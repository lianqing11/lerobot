# ManiSkill Integration Setup Summary

## 已完成的工作

成功在LeRobot中集成了ManiSkill环境支持，特别是StackCube-v1任务的评测功能。

### 1. 添加的文件

#### 核心集成文件

1. **`src/lerobot/envs/maniskill.py`** (新建)
   - ManiSkill环境包装器
   - 观察空间转换
   - 成功率跟踪
   - 环境工厂函数

2. **`src/lerobot/envs/configs.py`** (更新)
   - 添加了 `ManiSkillEnv` 配置类
   - 支持多种观察模式 (state, rgb, rgbd)
   - 支持多种控制模式
   - GPU/CPU 模拟支持

3. **`src/lerobot/envs/factory.py`** (更新)
   - 添加ManiSkill环境创建逻辑
   - 集成到环境工厂函数中

#### 测试和示例

4. **`test_maniskill_integration.py`** (新建)
   - 完整的集成测试脚本
   - 测试配置、环境创建、交互
   - ✅ 所有测试通过

5. **`examples/eval_maniskill_stackcube.sh`** (新建)
   - 评测脚本示例
   - 可直接运行

6. **`examples/train_maniskill_stackcube.sh`** (新建)
   - 训练脚本示例
   - 包含评测配置

7. **`MANISKILL_INTEGRATION.md`** (新建)
   - 完整的使用文档
   - 配置说明
   - 故障排除指南

### 2. 功能特性

✅ **支持的环境**
- StackCube-v1 (已测试)
- 所有ManiSkill任务 (理论支持)

✅ **支持的观察模式**
- `state`: 扁平状态向量
- `state_dict`: 结构化状态字典
- `rgb`: 视觉观察
- `rgbd`: 视觉+深度观察

✅ **支持的控制模式**
- `pd_ee_delta_pose`: 末端执行器增量控制
- `pd_joint_delta_pos`: 关节增量控制
- 其他ManiSkill支持的控制模式

✅ **仿真后端**
- CPU 仿真 (已测试)
- GPU 仿真 (支持，需要CUDA)

✅ **评测功能**
- 成功率跟踪
- 批量评测
- 并行环境
- 视频保存

## 使用方法

### 快速开始

#### 1. 测试集成

```bash
cd /root/projects/vla/lerobot
python test_maniskill_integration.py
```

#### 2. 评测已训练模型

```bash
# 使用示例脚本
bash examples/eval_maniskill_stackcube.sh /path/to/your/model

# 或者直接使用命令
lerobot-eval \
    --policy.path=your_model \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --eval.n_episodes=20
```

#### 3. 训练并评测

```bash
# 使用示例脚本
bash examples/train_maniskill_stackcube.sh /path/to/dataset

# 或者直接使用命令
lerobot-train \
    --dataset.repo_id=your_dataset \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --policy.type=act \
    --eval_freq=5000
```

### 配置示例

#### Python配置

```python
from lerobot.envs.configs import ManiSkillEnv

config = ManiSkillEnv(
    task="StackCube-v1",
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    fps=30,
    episode_length=50,
    sim_backend="cpu",
)
```

#### 命令行配置

```bash
lerobot-eval \
    --env.type=maniskill \
    --env.task=StackCube-v1 \
    --env.obs_mode=state \
    --env.control_mode=pd_ee_delta_pose \
    --env.fps=30 \
    --env.episode_length=50 \
    --env.sim_backend=cpu
```

## 技术细节

### 观察空间转换

ManiSkill的观察被包装成LeRobot期望的字典格式：

```python
# ManiSkill原始输出: array([...])
# 包装后输出: {"agent_pos": array([...])}
```

### 成功率跟踪

环境自动跟踪成功率：

```python
obs, reward, terminated, truncated, info = env.step(action)
success = info["success"]  # 布尔值
is_success = info["is_success"]  # 兼容性字段
```

### 向量化环境

支持多个并行环境：

```python
envs = make_env(config, n_envs=10, use_async_envs=True)
```

## 测试结果

```
================================================================================
✓ All tests PASSED!
================================================================================

测试项目：
✓ 环境配置创建
✓ 环境实例化
✓ 环境重置
✓ 环境交互
✓ 观察空间转换
✓ 动作空间兼容
✓ 成功率跟踪
```

## 性能建议

1. **GPU仿真** - 对于大规模评测：
   ```bash
   --env.sim_backend=gpu --env.num_envs=64
   ```

2. **异步环境** - 提高CPU利用率：
   ```bash
   --eval.use_async_envs=true
   ```

3. **批量评测** - 提高效率：
   ```bash
   --eval.batch_size=10 --eval.n_episodes=100
   ```

4. **评测频率** - 根据任务复杂度调整：
   - 简单任务: `--eval_freq=5000`
   - 复杂任务: `--eval_freq=10000`

## 支持的其他任务

除了StackCube-v1，还支持：

- **操作任务**: PickCube-v1, PegInsertionSide-v1, AssemblySquare-v1
- **灵巧操作**: TurnFaucet-v1, OpenCabinetDrawer-v1
- **移动操作**: PullCube-v1, PushCube-v1

使用方法相同，只需更改 `--env.task` 参数。

## 故障排除

### 问题1: 环境未找到

```
Error: Environment 'StackCube' doesn't exist
```

**解决**: ManiSkill会自动导入并注册环境，如果仍有问题，检查ManiSkill安装。

### 问题2: 观察维度不匹配

**解决**: 确保 `obs_mode` 与数据集一致：
- 数据集用state → 评测用 `--env.obs_mode=state`

### 问题3: 动作维度不匹配

**解决**: 确保 `control_mode` 在训练和评测时一致。

## 下一步

1. **测试其他任务**: 在不同的ManiSkill任务上测试集成
2. **GPU仿真测试**: 测试GPU仿真的性能提升
3. **多任务训练**: 设置多任务训练和评测
4. **性能优化**: 根据实际使用情况优化配置

## 文档参考

- 完整文档: `MANISKILL_INTEGRATION.md`
- 测试脚本: `test_maniskill_integration.py`
- 评测示例: `examples/eval_maniskill_stackcube.sh`
- 训练示例: `examples/train_maniskill_stackcube.sh`

---

**集成状态**: ✅ 完成并测试通过  
**最后更新**: 2025-10-21  
**测试环境**: StackCube-v1 with Panda robot

