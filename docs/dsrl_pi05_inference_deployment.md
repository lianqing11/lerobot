# Pi0.5 DSRL Latent-Noise Actor Inference Deployment

This document describes how to deploy the trained DSRL actor on another machine for real-robot RL rollout with Pi0.5.

The deployed model does **not** output robot actions directly. It outputs a 32-dimensional `latent_noise`. The deployment code repeats this `latent_noise` across the Pi0.5 chunk dimension to form `[50, 32]` noise, then passes that noise into Pi0.5 action generation.

## 1. What The Model Does

Training produced a lightweight actor-critic model:

```text
observation images + proprio -> actor -> latent_noise[32]
```

At inference time only the actor path is used:

```text
main camera image
wrist camera image
obs_state[13]
    -> frozen ResNet50 + proprio MLP + actor
    -> latent_noise[32]
    -> repeat to noise[50,32]
    -> Pi0.5 predict_action_chunk(..., noise=noise)
    -> robot action chunk[50, action_dim]
```

Terminology:

- `latent_noise`: the 32-dimensional SAC-trained noise.
- `noise`: the repeated Pi0.5 flow-matching noise with shape `[B, 50, 32]`.
- `action`: the real robot action produced by Pi0.5 and sent to the action queue.

Do not call the 32-dimensional model output an action. It is `latent_noise`.

## 2. Files To Copy To The Robot Machine

Copy these code files or the full repo:

```text
src/lerobot/rl/dsrl_pi05/model.py
src/lerobot/rl/dsrl_pi05/infer_noise.py
```

Copy the trained checkpoint, for example:

```text
ckpt/dsrl_pi05_1922_train_3000/checkpoint_001000.pt
```

The checkpoint contains:

```python
{
    "step": int,
    "args": {
        "obs_state_dim": 13,
        "latent_noise_dim": 32,
        "hidden_dim": ...,
        "feature_dim": ...,
        "noise_bound": 1.5,
        "resnet_weights": "imagenet",
        ...
    },
    "model": state_dict,
}
```

Use the same code version as training when possible. If code changed after training, verify that `LatentSAC` constructor arguments still match the checkpoint.

## 3. Runtime Dependencies

The robot machine needs:

```text
python
torch
torchvision
Pillow
numpy
lerobot repo / PYTHONPATH=src
Pi0.5 policy dependencies already used by your robot rollout code
```

If the checkpoint was trained with:

```text
resnet_weights = "imagenet"
```

then torchvision must be able to load `ResNet50_Weights.IMAGENET1K_V2`. The first run may download the weight file unless it is already cached at:

```text
~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
```

For deployment, preload/cache this file before running the robot to avoid network access during rollout.

## 4. Loading The Latent Actor

Use the helper in `infer_noise.py`:

```python
import torch

from lerobot.rl.dsrl_pi05.infer_noise import load_latent_sac, select_latent_noise, repeat_latent_noise

device = torch.device("cuda")
latent_sac = load_latent_sac("ckpt/dsrl_pi05_1922_train_3000/checkpoint_001000.pt", device=device)
latent_sac.eval()
```

Only the actor is used for action selection. The checkpoint also contains critics because they are needed for training and debugging.

## 5. Observation Preprocessing

The actor expects a batch dict with these keys:

```python
{
    "obs_state":   torch.float32, shape [B, 13],
    "image_main":  torch.float32, shape [B, 3, image_size, image_size],
    "image_wrist": torch.float32, shape [B, 3, image_size, image_size],
}
```

Images must be RGB and normalized with ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The training dataset used PIL bilinear resize. Match that in deployment:

```python
import numpy as np
import torch
from PIL import Image


def preprocess_rgb_image(image_rgb, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert RGB image array/PIL image to [1, 3, H, W] normalized tensor."""
    if not isinstance(image_rgb, Image.Image):
        image_rgb = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    image_rgb = image_rgb.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std
```

`obs_state` must match the training order:

```text
joint_1.pos
joint_2.pos
joint_3.pos
joint_4.pos
joint_5.pos
joint_6.pos
gripper.pos
ee_pos_x
ee_pos_y
ee_pos_z
ee_rot_rx
ee_rot_ry
ee_rot_rz
```

Use `float32`:

```python
obs_state = torch.tensor(obs_state_np, dtype=torch.float32, device=device).unsqueeze(0)
```

Do not min-max normalize `obs_state` unless the training code is changed to do the same. The current DSRL model was trained on the raw `obs_state` values stored in the sidecar.

## 6. Inference Flow

At every Pi0.5 decision point:

1. Capture the current robot observation.
2. Build the Pi0.5 policy batch as usual.
3. Build the DSRL actor batch from the same observation.
4. Actor outputs `latent_noise[1,32]`.
5. Repeat to `noise[1,50,32]`.
6. Call Pi0.5 action generation with this noise.
7. Send the resulting robot action chunk into the action queue.

Minimal code:

```python
import torch

from lerobot.rl.dsrl_pi05.infer_noise import select_latent_noise, repeat_latent_noise


@torch.no_grad()
def predict_pi05_chunk_with_dsrl(
    pi05_policy,
    latent_sac,
    pi05_batch,
    image_main_rgb,
    image_wrist_rgb,
    obs_state_np,
    image_size: int,
    device: torch.device,
    **pi05_kwargs,
):
    dsrl_batch = {
        "obs_state": torch.tensor(obs_state_np, dtype=torch.float32, device=device).unsqueeze(0),
        "image_main": preprocess_rgb_image(image_main_rgb, image_size, device),
        "image_wrist": preprocess_rgb_image(image_wrist_rgb, image_size, device),
    }

    latent_noise = select_latent_noise(latent_sac, dsrl_batch)  # [1, 32]
    noise = repeat_latent_noise(latent_noise, chunk_size=50)    # [1, 50, 32]

    action_chunk = pi05_policy.predict_action_chunk(
        pi05_batch,
        noise=noise.to(device),
        **pi05_kwargs,
    )
    return action_chunk, latent_noise, noise
```

For RTC/action queue deployment, keep passing the same runtime kwargs you currently use for Pi0.5:

```python
action_chunk = pi05_policy.predict_action_chunk(
    pi05_batch,
    noise=noise,
    inference_delay=inference_delay,
    prev_chunk_left_over=prev_chunk_left_over,
    execution_horizon=execution_horizon,
)
```

## 7. Action Queue Semantics

Pi0.5 produces a 50-step action chunk. The DSRL actor only selects the `latent_noise` used to generate that chunk.

If your queue executes only a prefix before replanning, keep the existing queue behavior. The DSRL actor should run at the same decision points where Pi0.5 previously sampled random noise.

Recommended deployment behavior:

```text
decision observation -> latent_noise[32]
latent_noise[32] -> repeat noise[50,32]
Pi0.5 -> action_chunk[50,7]
queue executes prefix / smoothed prefix as before
next decision -> compute new latent_noise
```

The rollout data must record which decision observation produced which `latent_noise`, because SAC training uses:

```text
(decision_observation, latent_noise, reward, next_decision_observation, done)
```

## 8. Rollout Data To Save

For every Pi0.5 decision/inference row, save at least:

```text
chunk_id
frame_index
task
obs_state
obs_image_main_jpeg
obs_image_wrist_jpeg
latent_noise                    # shape [32]
noise                           # optional, shape [50,32], should be repeat(latent_noise)
noise_shape                     # [50,32]
noise_shared                    # true
noise_scale                     # if used
action_chunk_raw                # [50, action_dim]
action_chunk_proc               # [50, action_dim]
action_chunk_shape              # [50, action_dim]
executed_start
executed_end
executed_count
task_success                    # terminal success label, broadcast to rows after episode ends
```

The existing sidecar format already stores most of this. The important DSRL-specific requirement is that `noise` must be shared/repeated across the 50 steps:

```python
assert noise.shape == (50, 32)
assert (noise == noise[0:1]).all()
```

Also save `latent_noise` explicitly if convenient. It avoids needing to recover it from `noise[0]`.

Reward:

- Use one terminal sparse reward per episode.
- `task_success=1` for success.
- `task_success=0` for failure.
- No frame reward is needed.

Training assigns the terminal reward to the final valid decision chunk and uses zero reward for the earlier chunks.

## 9. Safety Checks Before Robot Execution

Before sending actions to the robot:

```python
assert latent_noise.shape == (1, 32)
assert torch.isfinite(latent_noise).all()
assert noise.shape == (1, 50, 32)
assert torch.isfinite(noise).all()
```

Log these values per decision:

```text
latent_noise mean
latent_noise std
latent_noise min
latent_noise max
latent_noise abs max
```

If `abs max` is always close to `noise_bound` (`1.5` in current training), the actor may be saturating. Treat that checkpoint cautiously.

Recommended initial deployment gate:

- First run Pi0.5 base policy as before.
- Then run DSRL policy in a guarded critical phase only.
- Start with low number of episodes.
- Keep human stop/intervention available.
- Compare against the base Pi0.5 success rate and completion time.

## 10. Local Smoke Test On The Robot Machine

After copying the checkpoint and code, run:

```bash
PYTHONPATH=src python -m lerobot.rl.dsrl_pi05.infer_noise \
  --checkpoint ckpt/dsrl_pi05_1922_train_3000/checkpoint_001000.pt \
  --dataset-root /path/to/a/local/rollout_dataset \
  --index 0 \
  --device cuda \
  --image-size 128
```

Expected output:

```text
latent_noise shape=(1, 32) min=... max=...
repeated_noise shape=(1, 50, 32)
```

This command reads one sidecar observation from a local dataset and verifies that the actor checkpoint loads and produces the correct noise shape.

## 11. Common Failure Modes

### Shape mismatch loading checkpoint

The deployment `hidden_dim`, `feature_dim`, `obs_state_dim`, or `latent_noise_dim` does not match the checkpoint.

Use `load_latent_sac(...)`; it reads these values from checkpoint `args`.

### ResNet weight download during rollout

Pre-cache the ImageNet ResNet50 weight file before robot runs.

### Pi0.5 ignores the supplied noise

Verify that `pi05_policy.predict_action_chunk` passes `noise` through to `model.sample_actions`. In the current Pi0.5 implementation, `sample_actions(..., noise=noise)` uses the provided noise instead of sampling random noise.

### Image color/order mismatch

Training used RGB JPEG images decoded by PIL. If robot cameras provide BGR frames from OpenCV, convert to RGB before preprocessing:

```python
image_rgb = image_bgr[:, :, ::-1]
```

### Actor output saturates

If `latent_noise_abs_max` is always near `1.5`, try an earlier checkpoint or collect more online rollout data before relying on that checkpoint.

## 12. Recommended Metrics During Real-Robot RL Rollout

Record these per episode:

```text
episode_index
checkpoint_step
task_success
episode_duration_s
num_decision_chunks
num_executed_frames
mean_executed_count
latent_noise_abs_max_mean
latent_noise_abs_max_max
intervention_count
failure_reason
```

Track these over the latest 10-20 episodes:

```text
success_rate
median_episode_duration_s
successes_per_10_min
failure_count
```

Do not rely on actor loss or critic loss for deployment quality. Real robot success rate and completion time are the primary metrics.
