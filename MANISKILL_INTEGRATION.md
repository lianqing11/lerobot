# ManiSkill Integration with LeRobot

This document describes how to use ManiSkill environments with LeRobot for training and evaluation.

## Installation

First, ensure you have both LeRobot and ManiSkill installed:

```bash
# Install LeRobot
cd /root/projects/vla/lerobot
pip install -e .

# ManiSkill should already be installed
# If not: pip install mani-skill
```

## Quick Start

### 1. Evaluate a Trained Policy

Evaluate a pre-trained policy on the StackCube-v1 task:

```bash
lerobot-eval \
    --policy.path=your_trained_model \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --env.control_mode=pd_ee_delta_pose \
    --eval.n_episodes=20 \
    --eval.batch_size=10
```

### 2. Train a Policy with Evaluation

Train an ACT policy on StackCube-v1 with periodic evaluation:

```bash
lerobot-train \
    --dataset.repo_id=your_maniskill_dataset \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --env.control_mode=pd_ee_delta_pose \
    --policy.type=act \
    --steps=100000 \
    --eval_freq=5000 \
    --eval.n_episodes=10 \
    --batch_size=8 \
    --output_dir=outputs/maniskill_stackcube
```

## Configuration Options

### Environment Configuration

The `ManiSkillEnv` configuration class supports the following parameters:

```python
from lerobot.envs.configs import ManiSkillEnv

config = ManiSkillEnv(
    task="StackCube-v1",           # ManiSkill task name
    fps=30,                        # Control frequency
    episode_length=50,             # Maximum steps per episode
    obs_mode="state",              # Observation mode: "state", "rgb", "rgbd", "state_dict"
    control_mode="pd_ee_delta_pose",  # Control mode
    render_mode="rgb_array",       # Render mode
    sim_backend="cpu",             # "cpu" or "gpu"
    robot_uids="panda",            # Robot type
    camera_width=128,              # Camera width (for visual obs)
    camera_height=128,             # Camera height (for visual obs)
    num_envs=1,                    # Number of parallel envs (GPU only)
)
```

### Observation Modes

ManiSkill supports several observation modes:

- **`state`**: Flat state vector (robot joint positions)
  - Observation: `{"agent_pos": array([...])}`
  - Shape: `(48,)` for StackCube-v1 with Panda robot

- **`state_dict`**: Structured state dictionary
  - Includes robot state and task-specific extra information

- **`rgb`**: Visual observations from cameras
  - Observation: `{"base_camera": array([...])}`
  - Shape: `(height, width, 3)`

- **`rgbd`**: Visual + depth observations

### Control Modes

Common control modes:

- **`pd_ee_delta_pose`**: End-effector delta position/orientation control
  - Action dimension: 7 (3 position + 4 quaternion)

- **`pd_joint_delta_pos`**: Joint delta position control
  - Action dimension: depends on robot (8 for Panda)

## Supported Tasks

All ManiSkill tasks are supported. Some popular tasks:

- **Manipulation**: `StackCube-v1`, `PickCube-v1`, `PegInsertionSide-v1`
- **Dexterous**: `TurnFaucet-v1`, `OpenCabinetDrawer-v1`
- **Mobile**: `PullCube-v1`, `PushCube-v1`

## Example: Complete Training Pipeline

### Step 1: Prepare Dataset

Convert ManiSkill demonstrations to LeRobot format:

```bash
cd /root/projects/ManiSkill
python -m mani_skill.trajectory.convert_to_lerobot \
    trajectory.h5 \
    output_dir \
    --task-name "StackCube-v1" \
    --fps 30
```

### Step 2: Train with Evaluation

```bash
cd /root/projects/vla/lerobot

lerobot-train \
    --dataset.repo_id=output_dir \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --policy.type=act \
    --steps=100000 \
    --eval_freq=5000 \
    --eval.n_episodes=10 \
    --save_freq=10000 \
    --output_dir=outputs/stackcube_act
```

### Step 3: Evaluate Trained Policy

```bash
lerobot-eval \
    --policy.path=outputs/stackcube_act/checkpoints/050000/pretrained_model \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --eval.n_episodes=50 \
    --eval.batch_size=10
```

## GPU Simulation

For faster training/evaluation with GPU simulation:

```bash
lerobot-eval \
    --policy.path=your_model \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.sim_backend=gpu \
    --env.num_envs=64 \
    --eval.n_episodes=100 \
    --eval.batch_size=64
```

## Multi-Task Training

Train on multiple ManiSkill tasks (requires custom dataset setup):

```bash
lerobot-train \
    --dataset.repo_id=multitask_dataset \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --policy.type=act \
    --steps=200000
```

## Troubleshooting

### Issue: Environment not found

**Error**: `Environment 'StackCube' doesn't exist`

**Solution**: Make sure ManiSkill is properly installed and environments are registered:
```python
import mani_skill.envs  # This registers all environments
```

### Issue: Observation shape mismatch

**Problem**: The observation shape doesn't match the expected shape.

**Solution**: Check that the `obs_mode` in your config matches your dataset:
- Dataset created with `obs_mode="state"` → Use `env.obs_mode=state`
- Dataset with images → Use `env.obs_mode=rgb` or `rgbd`

### Issue: Action dimension mismatch

**Problem**: Action dimension doesn't match the policy output.

**Solution**: Ensure the `control_mode` matches between:
1. Dataset collection
2. Training configuration
3. Evaluation configuration

## Performance Tips

1. **Use GPU simulation** for faster parallel evaluation:
   ```bash
   --env.sim_backend=gpu --env.num_envs=64
   ```

2. **Adjust evaluation frequency** based on task complexity:
   - Simple tasks: `--eval_freq=5000`
   - Complex tasks: `--eval_freq=10000` or more

3. **Use async environments** for better CPU utilization:
   ```bash
   --eval.use_async_envs=true
   ```

4. **Batch evaluation** for efficiency:
   ```bash
   --eval.batch_size=10 --eval.n_episodes=100
   ```

## Integration Files

The ManiSkill integration consists of three main files:

1. **`src/lerobot/envs/configs.py`**: Contains `ManiSkillEnv` configuration class
2. **`src/lerobot/envs/maniskill.py`**: Environment wrapper and factory functions
3. **`src/lerobot/envs/factory.py`**: Updated to support ManiSkill environment creation

## Testing

Run the integration test to verify everything works:

```bash
cd /root/projects/vla/lerobot
python test_maniskill_integration.py
```

## Additional Resources

- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [ManiSkill GitHub](https://github.com/haosulab/ManiSkill)

