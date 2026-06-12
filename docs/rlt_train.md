# RLT Training Spec — Pi0.5 + ClawVLA Online RL

> 目标：在 bja2 上实现 RLT-style actor-critic 训练脚本，从 ClawVLA 的 dagger sidecar
> 数据中学一个 **直接输出 target action 的 Gaussian policy**，用于 Pi0.5 推理时
> 替换 reference action 的前 `C` 步。
>
> 参考论文：**RLT — Reinforced Latent Tokens / Reference-conditioned Training**
> (arXiv:2604.23073)。**不**使用 RL token / token-level critic；改用 RLT
> Appendix B 的标准 actor-critic（SAC）变体 + 一个 ResNet50 观察编码器。
>
> 推理端契约已在 `clawvla/rlt_actor.py` 落地，训练脚本必须严格对齐它的
> `RLTActorInterface` 接口和 `load_rlt_actor` 的 checkpoint 格式。

---

## 1. 概念定义（务必先读）

| 符号 / 名词 | 含义 |
|---|---|
| `Pi0.5` | 上游 VLA，输出 chunk_size=50、max_action_dim=32 的 raw action |
| `postprocessor` | Pi0.5 后处理器，把 raw 32-d → 物理 7-d，输出 `[50, 7]` |
| `ã` | **VLA reference action**：Pi0.5 postprocessor 输出，物理单位 `[50, 7]` |
| `a` | **target action**：RLT actor 直接输出（**非 residual**），物理单位 `[C, 7]` |
| `executed` | 真正下发到机器人的动作 `[50, 7]`：前 `C` 步为 `a`，后面回退 `ã` |
| `C` (`chunk_C`) | actor 决策的步数，超参，要满足 `1 ≤ C ≤ 50`。建议 `C = 16` |
| `σ` (`action_std`) | actor Gaussian 的 **固定标准差** —— 训练时手动设置、不学习 |
| `r` | sparse terminal reward：episode 成功 = 1，失败 = 0 |

**核心 RL formulation**：

```
π(a | obs, ã) = N(μθ(obs, ã), σ²·I)         # Eq. (4)
a = μθ + σ·ε,  ε ~ N(0, I)                  # 探索时
a = μθ                                       # eval / deployment
```

`σ` 不学，不需要 entropy bonus。critic 仍用 SAC 双 Q + target network。

---

## 2. 数据来源

### 2.1 目录结构（ClawVLA 端）

每个 dagger 数据集形如：

```
datasets/<repo_name>/
  meta/episodes.jsonl                # 含每 ep 的 task_success
  data/chunk-000/episode_000000.parquet   # LeRobotDataset 主表
  videos/...
  inferences/episode_000000.parquet  # ← sidecar：训练只读这个
```

**训练只读 `inferences/` 下的 parquet 即可。**
主 LeRobotDataset 仅用于：
1. 拿 `task_success`（也可以直接广播到 sidecar，见 §2.3）
2. （可选）拿稠密观察做调试

### 2.2 Sidecar 列定义（每行 = 一次 Pi0.5 推理）

由 `clawvla/data_collect/run_dagger.py::InferenceRecorder` 产出。

| 列 | dtype | shape | 含义 |
|---|---|---|---|
| `chunk_id` | int | scalar | episode 内单调递增（0,1,2,...） |
| `frame_index` | int | scalar | 主表对应帧索引 |
| `task` | str | scalar | 自然语言指令 |
| `obs_state` | float32 list | `[13]` | proprio：6 joint + gripper + 3 ee_pos + 3 ee_rot |
| `obs_image_main_jpeg` | bytes | — | 决策时刻主相机 JPEG |
| `obs_image_wrist_jpeg` | bytes | — | 决策时刻腕部相机 JPEG |
| `noise` | float32 list | `[50, 32]` flatten | Pi0.5 flow 噪声（RLT 模式逐步独立） |
| `noise_shape` | int list | `[50, 32]` | — |
| `noise_scale` | float | scalar | RLT 模式恒为 `1.0`；SAC 模式 `-1.0` |
| `noise_shared` | bool | scalar | RLT 模式 `False`；SAC 模式 `True` |
| `action_chunk_raw` | float32 list | `[50, 32]` flatten | Pi0.5 raw 输出（normalized） |
| `action_chunk_proc` | float32 list | `[50, 7]` flatten | **executed action**（物理单位） |
| `action_chunk_shape` | int list | `[50, 7]` | — |
| `vla_reference_chunk` | float32 list | `[50, 7]` flatten | **ã**（物理单位，未经 actor / warmup_noise 修改） |
| `actor_ckpt_step` | int | scalar | 产出该行的 actor 训练步数；`-1` 表示无 actor |
| `actor_action_std` | float | scalar | 见下表 |
| `executed_start` | int | scalar | 该 chunk 在 robot 时间轴上**开始**执行的步号 |
| `executed_end` | int | scalar | 该 chunk **结束**执行的步号 |
| `executed_count` | int | scalar | `= executed_end - executed_start`，被实际执行的步数 |
| `prev_left_over_len` | int | scalar | 上一 chunk 残留长度 |
| `inference_delay` / `real_delay` | int | scalar | RTC 延迟（帧） |
| `task_success` | int | scalar | episode 结束后 S/F 打标，**广播到所有行** |

#### `actor_action_std` 编码（三类来源数据）

| `actor_ckpt_step` | `actor_action_std` | 数据语义 |
|---|---|---|
| `-1` | `-1.0` | **Pure warmup**：`executed = ã`，纯 VLA |
| `-1` | `> 0`（如 `0.05`） | **Warmup with noise**：`executed = ã + σ·ε`，σ 由 `--warmup-noise` 设定 |
| `≥ 0` | `> 0` 或 `0.0` | **Online RL**：`executed[:C] = actor.select(...)`；`0.0` 代表 `--rlt-deterministic` 用 μ 不加 σ |

### 2.3 Task success 广播

`task_success` 在每个 sidecar 文件 **保存时已经写入每一行**（同 episode 的所有行
同值），无需额外去主表读取。直接用就行。

---

## 3. RL Transition 构造

每个 sidecar 行 = 一个 transition。**只用前 `C` 步**：

```python
obs_t       = (obs_image_main_jpeg, obs_image_wrist_jpeg, obs_state)   # 该 row
ref_t       = vla_reference_chunk[:C, :]                               # [C, 7]
action_t    = action_chunk_proc[:C, :]                                 # [C, 7]
reward_t    = task_success if is_last_chunk_in_episode else 0.0
done_t      = is_last_chunk_in_episode
obs_tp1     = (next row's obs)        # 同 episode 下一个 row；末 row 用 zeros 占位
ref_tp1     = next row's vla_reference_chunk[:C, :]
```

**"last chunk" 判定**：episode 内 `chunk_id` 最大的行。

`reward_t` 由 `task_success ∈ {0, 1}`（失败/成功）线性映射，**不要**人工加 shape。

---

## 4. 网络架构

### 4.1 共享观察编码器（与 DSRL 对齐）

复用 bja2 上 `lerobot/rl/dsrl_pi05/model.py` 的：

```text
SharedResNet50Encoder      # frozen IMAGENET1K_V2，输出 2048-d，两路 cat → 4096-d
ObservationEncoder         # image(4096) + proprio_mlp(13→256) → fusion_mlp → feature_dim
```

默认超参（**checkpoint args 里要记录**，供推理端 reconstruct）：

```python
obs_state_dim     = 13
proprio_hidden    = 256
feature_dim       = 256              # ← RLT 用 256，跟 DSRL 的 512 不同
resnet_weights    = "imagenet"       # frozen
image_size        = 128
```

`ObservationEncoder` 直接从 DSRL `model.py` import 即可。

### 4.2 RLT Actor

```text
input  : feature[256] || ref_flat[C·7]
         → mlp(hidden_layers=3, hidden_dim=256, activation=Tanh+LayerNorm)
         → μ[C·7]
output : reshape → μ[C, 7]
```

**强制要点**：
1. **没有 log_std 头**：σ 是 `--action-std` 超参，恒定。actor MLP 只产 μ。
2. **Reference dropout**：训练时按 50% 概率把 `ref_flat` 全部置零，强迫 actor 不能
   只复制 ã。inference 时永远传入真实 ã。
3. 输出物理单位，不做 tanh squash（动作范围由 Pi0.5 后处理保证，actor 学到的是
   reference 附近的小扰动）。

### 4.3 Critic（双 Q）

```text
input  : feature[256] || action_flat[C·7]
         → mlp(hidden_layers=3, hidden_dim=256, activation=Tanh+LayerNorm)
         → Q ∈ ℝ
```

两个独立 Critic + 各自 target network。**Critic 接 `action`，不接 `ã`** —— 因为
Q 评估的是 executed action 的回报。

### 4.4 总模块 `RLTPolicy`

```python
class RLTPolicy(nn.Module):
    encoder:        ObservationEncoder         # shared
    actor:          RLTActor
    critic1:        RLTCritic
    critic2:        RLTCritic
    target_critic1: RLTCritic
    target_critic2: RLTCritic
    # 注意：没有 log_alpha，没有 alpha optimizer
```

state_dict key 前缀必须严格是 `encoder.*` / `actor.*` / `critic1.*` / `critic2.*` /
`target_critic1.*` / `target_critic2.*`，推理端 `clawvla/rlt_actor.py` 会以
`strict=True` 加载。

---

## 5. 训练算法（SAC-style with fixed σ）

每个 update step：

1. 从 replay 采 batch（B=256 起）`(obs, ref, action, reward, done, obs', ref')`
2. 编码 `feature = encoder(obs)`, `feature' = encoder(obs')`（**target Q 用 no_grad 的 encoder**，与 DSRL 同步）
3. 计算 target Q：
   ```
   μ' = actor(feature', ref')                    # 注意 inference 用 μ
   a' = μ' + σ · ε,  ε ~ N(0, I)                 # 与 rollout 一致
   q' = min(target_critic1(feature', a'), target_critic2(feature', a'))
   y  = reward + γ · (1 - done) · q'             # 无 entropy 项
   ```
4. Critic loss：MSE(critic_i(feature, action), y) for i in {1,2}
5. Actor loss（**确定性策略梯度风格**）：
   ```
   μ      = actor(feature, ref)                  # ref 这里以 50% 概率被 dropout
   q_pi   = min(critic1(feature.detach(), μ), critic2(feature.detach(), μ))
   loss_a = -q_pi.mean() + λ_bc · MSE(μ, action)    # 可选 BC anchor
   ```
6. Target update：Polyak `τ=0.005`
7. **No log_alpha update**，**no entropy bonus**。

### 5.1 BC Anchor（可选但推荐）

warmup 阶段（前 5k step）`λ_bc = 1.0`，之后线性退到 `0.01`。BC target 用
`action` 而非 `ref` —— 这样在 warmup-with-noise 数据上 actor 学到的是带探索的均值。

### 5.2 默认超参

```yaml
batch_size:      256
gamma:           0.99
tau:             0.005
lr_actor:        3e-4
lr_critic:       3e-4
action_std:      0.05         # 物理单位 σ，约等于 0.5° per joint step
chunk_C:         16
ref_dropout_p:   0.5
bc_lambda_init:  1.0
bc_lambda_final: 0.01
bc_anneal_steps: 5000
total_steps:     50000
log_every:       100
ckpt_every:      1000
warmup_steps:    1000         # 期间只更新 critic，不更新 actor
```

---

## 6. Dataloader

参考 bja2 `lerobot/rl/dsrl_pi05/dataset.py` 的实现，按 transition 索引：

1. **启动时扫描**：递归扫所有 `datasets/*/inferences/episode_*.parquet`，
   收集 `(file_path, row_idx, episode_chunk_count)`。
2. **过滤**：丢弃 `task_success == -1`（未打标）的 episode。
3. **__getitem__**：
   - 读出 row 的图像（JPEG decode → resize 128 → ImageNet normalize）
   - `obs_state` 直接 `torch.as_tensor`
   - `ref` = `vla_reference_chunk[:C]`
   - `action` = `action_chunk_proc[:C]`
   - `reward` / `done`：见 §3
   - `next_obs` / `next_ref`：取同 episode 下一行；若是末行，则 `done=True`、
     `next_obs/next_ref` 用 zeros 占位（target Q 会被 `(1 - done)` 清零）
4. **数据均衡（可选）**：对 `task_success==1` 的 episode 上采样，避免成功率低导致
   reward 信号稀疏。

### 6.1 图像预处理

直接复用 `clawvla/dsrl_actor.py::preprocess_rgb_image`（已经在仓库里），输入是
RGB ndarray 或 PIL，输出 `[1, 3, 128, 128]` ImageNet-normalized。

JPEG 解码用 `cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]`
（BGR→RGB）。

---

## 7. Checkpoint 格式（**必须严格对齐**）

`clawvla/rlt_actor.py::load_rlt_actor` 期望：

```python
{
    "step": int,                  # 训练步数
    "args": {
        "obs_state_dim":   13,
        "proprio_hidden":  256,
        "feature_dim":     256,
        "resnet_weights":  "imagenet",
        "image_size":      128,
        "chunk_C":         16,
        "action_dim":      7,
        "action_std":      0.05,
        # ... 其它训练超参也保留进来便于复现
    },
    "model": state_dict,          # 整个 RLTPolicy.state_dict()
}
```

**state_dict 命名约束**：

```
encoder.image_encoder.backbone.*    # frozen ResNet50
encoder.proprio_encoder.*
encoder.fusion.*
actor.net.*
critic1.net.*
critic2.net.*
target_critic1.net.*
target_critic2.net.*
```

推理端只会读 `encoder.*` 和 `actor.*`，critic 写进 ckpt 是为了能 resume 训练。

---

## 8. 推理端契约对齐

`clawvla/rlt_actor.py::RLTActorInterface` 定义如下（**不要改**）：

```python
class RLTActorInterface(Protocol):
    chunk_C:    int
    action_dim: int
    image_size: int
    ckpt_step:  int
    action_std: float

    @torch.no_grad()
    def select(
        self,
        batch: dict[str, Tensor],   # {obs_state, image_main, image_wrist}
        ref_chunk_phys: Tensor,     # [1, chunk_C, action_dim]
        *,
        deterministic: bool = False,
    ) -> Tensor:                    # 返回 [1, chunk_C, action_dim] 物理单位
        ...
```

`load_rlt_actor(checkpoint_path, device)` 必须：

1. `torch.load(ckpt, map_location=device, weights_only=False)`
2. 用 `ckpt["args"]` 构造 `RLTPolicy`
3. `model.load_state_dict(ckpt["model"], strict=True)`
4. 调用 `model.eval()`
5. **包一层 thin adapter**，使其满足 `RLTActorInterface`：

```python
class _RLTActorAdapter:
    def __init__(self, policy: RLTPolicy, ckpt_args: dict, ckpt_step: int):
        self.policy     = policy
        self.chunk_C    = int(ckpt_args["chunk_C"])
        self.action_dim = int(ckpt_args["action_dim"])
        self.image_size = int(ckpt_args["image_size"])
        self.ckpt_step  = int(ckpt_step)
        self.action_std = float(ckpt_args["action_std"])

    @torch.no_grad()
    def select(self, batch, ref_chunk_phys, *, deterministic: bool = False):
        feature = self.policy.encoder(
            batch["obs_state"], batch["image_main"], batch["image_wrist"],
        )
        ref_flat = ref_chunk_phys.reshape(ref_chunk_phys.shape[0], -1)
        mu = self.policy.actor(feature, ref_flat)               # [1, C*action_dim]
        mu = mu.view(-1, self.chunk_C, self.action_dim)
        if deterministic:
            return mu
        eps = torch.randn_like(mu)
        return mu + self.action_std * eps
```

**这段 adapter 实现要落到 `clawvla/rlt_actor.py` 里**（同时实现 `load_rlt_actor`）。
建议训练脚本结束后，**把 bja2 训练代码的 `RLTPolicy / RLTActor / ObservationEncoder`
精简版以 inference-only 形式 port 进 `clawvla/rlt_actor.py`** —— 跟 DSRL 那套
（`clawvla/dsrl_actor.py` 镜像 `lerobot/rl/dsrl_pi05/model.py`）做法一致。

---

## 9. 训练脚本 CLI

```bash
python -m lerobot.rl.rlt_pi05.train \
    --datasets-root /VLA-Data/.../datasets \
    --include "paper_ball_*"  \
    --out-dir   ckpt/rlt_paper_ball_round1 \
    --chunk-C   16 \
    --action-std 0.05 \
    --batch-size 256 \
    --total-steps 50000 \
    --ckpt-every  1000
```

`--include` 是 glob 匹配数据集目录名；可多次出现。

---

## 10. 建议目录结构（bja2 端）

```
src/lerobot/rl/rlt_pi05/
    __init__.py
    dataset.py        # RLTSidecarDataset + RLTReplayBuffer
    model.py          # ObservationEncoder (可 import 自 dsrl_pi05) + RLTActor + RLTCritic + RLTPolicy
    train.py          # 主训练循环，CLI 入口
    infer.py          # 仅做 smoke：load ckpt 跑一个 batch 检查 forward
```

---

## 11. 验证 checklist（实现后 codex 必须自测）

1. `python -m lerobot.rl.rlt_pi05.train --dry-run`：不更新参数，跑一遍 dataloader
   + forward + 算 loss，确认无 shape mismatch。
2. 在一个 dataset 上跑 100 step，确认：
   - critic loss 在下降
   - actor 输出 μ 的 norm 与 ã 的 norm 同量级
   - ref_dropout 关闭时 μ 与 ã 的 cos sim ≥ 0.9（actor 应能 copy ã）
   - ref_dropout 开启时 μ ≠ ã（actor 不能完全依赖 ref）
3. 存 ckpt → scp 到 ClawVLA → 在本地用
   ```python
   from clawvla.rlt_actor import load_rlt_actor
   actor = load_rlt_actor("path/to.pt", device="cuda")
   ```
   能 strict=True 加载并 forward 出 `[1, C, 7]` 输出。
4. 在 ClawVLA 本地用 `bash clawvla/data_collect/run_dagger_raw.sh ... --rlt-model
   path/to.pt --rlt-deterministic` 起一个 L 模式 episode，确认推理 log 里
   `[RLT] chunk=... step=... σ=...` 正常打印。

---

## 12. 不做的事

- ❌ **RL token / Q-token 注意力机制**：复杂度不值，且我们没有 LLM 端 token 接入
- ❌ **学习 σ**：固定就好；调 σ 是采集端事
- ❌ **Entropy bonus / alpha tuning**：σ 固定后没意义
- ❌ **Action squash (tanh)**：物理单位下不需要
- ❌ **多任务条件输入到 actor**：当前 actor 不接 `task` 字符串，靠 ResNet 特征
- ❌ **稠密 reward shape**：只用 sparse terminal `task_success`

---

## 13. 与 ClawVLA 端的协议

```
ClawVLA 端：
    - 推理 schema 由 clawvla/rlt_actor.py::RLTActorInterface 定义（已落地）
    - sidecar 数据 schema 由 InferenceRecorder 决定（已落地，见 §2.2）
    - 训练 ckpt 文件由 codex 在 bja2 上产出，scp 到本地后用
      load_rlt_actor() 加载

bja2 端：
    - 实现 src/lerobot/rl/rlt_pi05/{dataset,model,train,infer}.py
    - 输出 ckpt 格式严格按 §7 写
    - 训练完后 port 一份 inference-only 模块过 PR 到 ClawVLA 仓库
      的 clawvla/rlt_actor.py（替换当前的 NotImplementedError stub）
```

---

## 14. 当前已采集数据备注

- 任务：paper-ball-into-trash-bin
- Pi0.5 base ckpt：`lianqing_0319_full_train_14k`（noise 敏感性较好）
- Warmup 模式：L 键 + （可选）`--warmup-noise 0.05`
- 每条 sidecar 行的 `noise_shared = False`、`noise_scale = 1.0` 可用来识别 RLT 数据
