# SAC Online RL 数据集格式

> 由 `run_dagger.py` 的 `sac_online_rl` 模式产出。用于 Pi0.5 (flow-matching) 的离线/在线 SAC 训练。

---

## 0. 采集方式

在 `run_dagger.py` 启动后，每个 episode 开始前的菜单里按 **`R`** 触发 `sac_online_rl` 模式：
- 行为同 `policy_eval`：全程 Policy 自主跑，结束后人工按 `S` (成功) / `F` (失败) 打标
- 唯一区别：开 `InferenceRecorder` 旁路写入推理 sidecar

每条 episode 产出：
1. 标准 LeRobotDataset 主表 + 视频（执行时刻 30Hz 流）
2. **新增** `inferences/episode_{nnnnnn}.parquet` 推理 sidecar（变速 ~1.5Hz，每次推理一行）

---

## 1. 目录布局

```
datasets/{repo_id}/
├── data/chunk-000/
│   └── episode_{nnnnnn}.parquet      # 主表 (30Hz)
├── videos/chunk-000/observation.images.{cam}/
│   └── episode_{nnnnnn}.mp4          # 执行时刻相机视频流 (30Hz)
├── meta/
│   ├── info.json                     # LeRobot 标准
│   ├── episodes/                     # episode 级元数据 (mode, reward, num_chunks)
│   ├── tasks.jsonl
│   └── stats.json
└── inferences/
    └── episode_{nnnnnn}.parquet      # 推理 sidecar (~1.5Hz, 每次推理一行)
```

`{nnnnnn}` 是 6 位数 episode_index，零填充。主表与 sidecar 文件名一一对应。

---

## 2. 主表 schema（LeRobotDataset 标准 + 新增 2 列）

每行 = 30Hz 控制循环里实际发给机械臂的一帧。

| 列 | 类型 | 含义 |
|---|---|---|
| `observation.state` | float32[N_joints] | 执行时刻关节角（degrees） |
| `observation.images.{cam}` | video (HxWx3 uint8 BGR) | 执行时刻相机帧，存在 mp4 里 |
| `action.{joint}.pos` 等 | float32 | 实际发给机械臂的动作 |
| `task` | str | 自然语言指令 |
| `intervention` | int8[1] | DAgger 用：1=人工 / 0=Policy / -1=N/A（SAC 模式恒为 -1） |
| **`chunk_id`** | int32[1] | 该 action 来自的推理调用 id；`-1` 表示推理还没产出（episode 开头） |
| **`chunk_offset`** | int32[1] | action 在原始 chunk 中的索引（已含 `real_delay`） |
| `timestamp` | float64 | 自 episode 起的秒数 |
| `frame_index` | int64 | episode 内单调递增 |
| `episode_index` | int64 | dataset 内 episode 编号 |

**通过 `(episode_index, chunk_id)` 可以从主表 join 到 sidecar。**

---

## 3. Sidecar schema（`inferences/episode_{nnnnnn}.parquet`）

每行 = 一次 `policy.predict_action_chunk` 调用。一个 60s episode 通常有 ~90 行。

| 列 | 类型 | shape (flatten 后元素数) | 含义 |
|---|---|---|---|
| `chunk_id` | int64 | scalar | episode 内单调递增 |
| `t_inference_start_s` | float64 | scalar | 相对 episode 起始的秒数 |
| `frame_index` | int64 | scalar | 推理开始时主表的当前 frame_index |
| `task` | string | scalar | 自然语言指令 |
| `inference_delay` | int64 | scalar | 传给 RTC 的预测延迟（帧） |
| `real_delay` | int64 | scalar | 推理结束时实测的延迟（帧）；merge 时丢掉 chunk 前 `real_delay` 帧 |
| `prev_left_over_len` | int64 | scalar | 上一次 chunk 进入本次推理时残留的动作数（RTC 的 `prev_chunk_left_over`） |
| `executed_start` | int64 | scalar | = `real_delay`，本 chunk 中首个被实际执行的索引 |
| `executed_end` | int64 | scalar | 末个被执行的索引 + 1（下次 merge 时回填，最后一个 chunk 在 episode 结束时 finalize） |
| `executed_count` | int64 | scalar | = `executed_end - executed_start`，实际执行了几帧 |
| `obs_state` | list&lt;float&gt; | 6 | 推理时刻的原始关节角（degrees） |
| `noise` | list&lt;float&gt; | 50×32 = 1600 | flow-matching 初始噪声 $x_0 \sim N(0,1)$ |
| `noise_shape` | list&lt;int&gt; | [50, 32] | [chunk_size, max_action_dim] |
| `action_chunk_raw` | list&lt;float&gt; | 50×7 = 350 | 模型输出（去 padding 后的真实 action 维度） |
| `action_chunk_proc` | list&lt;float&gt; | 50×7 = 350 | 经 postprocessor 反归一化后的 chunk（与发给机械臂的一致） |
| `action_chunk_shape` | list&lt;int&gt; | [50, 7] | [chunk_size, action_dim] |
| `obs_image_{cam}_jpeg` | binary | — | 推理时刻每路相机的 JPEG 字节（quality=90），cam 与主表 `observation.images.{cam}` 同名 |
| `task_success` | int64 | scalar | episode 级 reward，**广播到每一行**：`1`=成功 / `0`=失败 / `-1`=未打标 |

**关键关系**：

```
对一条 sidecar 行：
  本次推理生成 chunk_size=50 个 action
  实际执行的子区间为 chunk[executed_start : executed_end]   长度 = executed_count
  对应的主表行：frame_index ∈ [chunk_start_frame, chunk_start_frame + executed_count)
  这些主表行的 chunk_id = 本行 chunk_id
  这些主表行的 chunk_offset ∈ [executed_start, executed_end)
```

> 注：由于 TCS 平滑 patch，主表中实际下发的 action 在 chunk 边界 8 帧内是新旧 chunk 的线性混合，**不严格等于** `action_chunk_proc[chunk_offset]`。模型本身真正"决定"的 action 是 `action_chunk_raw`。

---

## 4. Episode 级元数据（`meta/episodes/*.parquet`）

`episodes.parquet` 标准列之外，本模式额外写入：

| 列 | 类型 | 含义 |
|---|---|---|
| `episode_mode` | string | `"sac_online_rl"` (固定值，可借此过滤 SAC episodes) |
| `task_success` | int | **sparse reward**：`1`=成功 / `0`=失败 / `-1`=未打标（异常退出时） |
| `num_chunks` | int | 该 episode 的推理调用数（= sidecar 行数） |

> **`task_success` 在 sidecar 的每一行也有一份**（广播值），所以单独读 sidecar 就能拿到 reward，无需再 join episode metadata。两份数据保证一致。

---

## 5. 读取方式（参考实现）

### 5.1 加载一个 episode 的 sidecar

```python
import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

def load_episode_chunks(dataset_root: str, episode_index: int) -> list[dict]:
    """读取 sidecar，返回每个 chunk 的字典列表。"""
    path = f"{dataset_root}/inferences/episode_{episode_index:06d}.parquet"
    table = pq.read_table(path)

    chunks = []
    for row in table.to_pylist():
        chunk_size, action_dim = row["action_chunk_shape"]
        c, d = row["noise_shape"]

        chunk = {
            "chunk_id":          row["chunk_id"],
            "t_start_s":         row["t_inference_start_s"],
            "frame_index":       row["frame_index"],
            "task":              row["task"],
            "real_delay":        row["real_delay"],
            "executed_start":    row["executed_start"],
            "executed_end":      row["executed_end"],
            "executed_count":    row["executed_count"],
            "task_success":      row["task_success"],                                # 1 / 0 / -1
            "obs_state":         np.array(row["obs_state"], dtype=np.float32),       # (6,)
            "noise":             np.array(row["noise"], dtype=np.float32).reshape(c, d),         # (50, 32)
            "action_raw":        np.array(row["action_chunk_raw"], dtype=np.float32).reshape(chunk_size, action_dim),    # (50, 7)
            "action_proc":       np.array(row["action_chunk_proc"], dtype=np.float32).reshape(chunk_size, action_dim),   # (50, 7)
            "images":            {},
        }
        for col in table.column_names:
            if col.startswith("obs_image_") and col.endswith("_jpeg"):
                cam = col[len("obs_image_"):-len("_jpeg")]
                jpeg_bytes = row[col]
                if jpeg_bytes is not None:
                    chunk["images"][cam] = np.array(Image.open(io.BytesIO(jpeg_bytes)))
        chunks.append(chunk)
    return chunks
```

### 5.2 读取 episode 级 reward + mode

```python
import pandas as pd

def load_episode_meta(dataset_root: str) -> pd.DataFrame:
    """返回 episodes 元数据，含 episode_mode / task_success / num_chunks。"""
    import glob
    files = sorted(glob.glob(f"{dataset_root}/meta/episodes/**/*.parquet", recursive=True))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
```

### 5.3 join 主表 ↔ sidecar（可选）

如果你需要"在 chunk 的某个 offset 处真实下发的 action"以及对应的执行时刻图像：

```python
import pyarrow.parquet as pq
main_table = pq.read_table(f"{dataset_root}/data/chunk-000/episode_{ep:06d}.parquet").to_pandas()
sidecar = load_episode_chunks(dataset_root, ep)

# 给定一个 chunk
chunk = sidecar[0]
mask = (main_table["chunk_id"] == chunk["chunk_id"]).to_numpy().squeeze()
executed_rows = main_table[mask]
# executed_rows 包含 chunk_offset ∈ [executed_start, executed_end)
# 视频帧可通过 frame_index + decord/ffmpeg 解出来
```

---

## 6. SAC 训练用的最小 dataloader 形态

典型 SAC transition：`(s_t, a_t, r_t, s_{t+1}, done)`。对 chunk-level SAC：

```python
import torch
from torch.utils.data import Dataset

class SACChunkDataset(Dataset):
    """
    一个 sample = 一次推理调用。
      observation: image(s) + proprio(state) + task
      action:      整个 chunk (50, 7) 或被截断到 executed_count
      noise:       生成该 chunk 的初始噪声 (50, 32)
      reward:      episode 级 sparse reward 广播给最后一个 chunk；其他 chunk reward=0
      done:        最后一个 chunk = True
    """
    def __init__(self, dataset_root: str, mode_filter: str = "sac_online_rl"):
        self.root = dataset_root
        meta = load_episode_meta(dataset_root)
        meta = meta[meta["episode_mode"] == mode_filter].reset_index(drop=True)

        self.index = []  # (episode_index, chunk_idx_in_ep, is_last, reward)
        for _, row in meta.iterrows():
            ep = int(row["episode_index"])
            n = int(row["num_chunks"])
            r = int(row["task_success"])  # 1 / 0 / -1
            for i in range(n):
                self.index.append((ep, i, i == n - 1, r))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep, ci, is_last, r = self.index[idx]
        chunks = load_episode_chunks(self.root, ep)
        c = chunks[ci]
        reward = float(r) if (is_last and r >= 0) else 0.0
        return {
            "obs_state":  torch.from_numpy(c["obs_state"]),                  # (6,)
            "obs_images": {k: torch.from_numpy(v) for k, v in c["images"].items()},  # HWC uint8
            "task":       c["task"],
            "noise":      torch.from_numpy(c["noise"]),                      # (50, 32)
            "action":     torch.from_numpy(c["action_raw"][c["executed_start"]:c["executed_end"]]),  # (executed_count, 7)
            "action_full": torch.from_numpy(c["action_raw"]),                # (50, 7), 含未执行的尾巴
            "reward":     torch.tensor(reward, dtype=torch.float32),
            "done":       torch.tensor(is_last, dtype=torch.bool),
        }
```

**生产环境建议**：
- 把 `load_episode_chunks` 加 LRU 缓存（同 episode 的 90 个 chunk 同源）
- 主表 RGB 图像如果需要，用 `decord.VideoReader` 按 `frame_index` 取帧
- 失败的 episode（`reward == 0`）和成功的（`reward == 1`）按需做 class-balance 采样

---

## 7. 老数据回填 `task_success`

对于在我加 `task_success` 列**之前**录的 sidecar 文件（缺该列但 `meta/episodes/*.parquet` 里已有 tag），用回填脚本一次性补齐：

```bash
python -m clawvla.data_collect.backfill_sidecar_reward \
    /path/to/datasets/dagger-20260526-1730-task03-...
```

特性：
- 自动从 `meta/episodes/*.parquet` 读 `(episode_index → task_success)` 映射
- 已经有 `task_success` 列的文件自动 skip（幂等）
- 支持多个数据集：`backfill_sidecar_reward ds1 ds2 ds3`
- 支持手动覆盖：`--override 0=1 --override 1=0`（episode 0 标成功、episode 1 标失败）

## 8. 字段速查（最小集）

如果只用最核心的字段做 RL 训练：

| 目的 | 字段 |
|---|---|
| Observation (policy 输入复现) | sidecar.`obs_image_{cam}_jpeg` + sidecar.`obs_state` + sidecar.`task` |
| Policy 决策的"action" | sidecar.`action_chunk_raw` (模型直出，与 noise 配对) |
| 实际执行了哪些 | sidecar.`executed_start` / `executed_end` |
| 生成 chunk 的随机性 | sidecar.`noise` |
| Episode reward | sidecar.`task_success`（每行广播；meta/episodes.parquet 也有一份） |
