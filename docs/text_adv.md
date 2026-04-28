# Advantage-conditioned RL with Text Indicator

## 1. 工作核心

baseline 的做法是：先用 reward 训练 `value / progress estimator`，再把每个动作压成一个 `binary advantage indicator`，最后训练 `pi(a | o, task, indicator)`。它的价值是让 policy 能利用 mixed-quality data，并通过 conditional imitation learning 做 policy improvement。

我们想优化的不是整个 RL 框架，而是它的 condition 接口。binary indicator 太粗：训练时同一个 label 下仍然混着很多动作模式，推理时也只能表达“朝高 advantage 采样”，不能更具体地描述当前应该生成的局部动作。

因此，我们的目标是把 `binary advantage indicator` 改成 `text condition`：

- 训练时，用更细粒度的 `positive / negative` 动作模式文本做 supervision
- 推理时，只输入正向 guidance text，让 policy 朝期望的动作模式采样

## 2. 核心假设

- 训练侧：如果用动作模式文本替代 binary indicator，那么同一 condition 下的动作分布会更集中，conditional imitation learning 更容易学到清晰的条件到动作映射。
- 推理侧：如果给 policy 的不是抽象的 `true`，而是一句正向的动作模式描述，那么动作生成会更可控。

这里一个关键前提是：guidance 的粒度不应该是整个 episode，也不应该粗略绑定整个 stage，而应该尽量对应当前局部动作单元，例如 `action-chunk-level` 或 `local transition-level`。

## 3. Text 形式

这里的 text 不是状态报告，也不是字段硬拼接，而是动作模式描述。

训练时示例：

```text
Positive: align with the bottle center and secure a stable grasp.
Negative: grasp drifts to the bottle side, contact is unstable, and the bottle is not securely held.
```

中文示例：

```text
Positive: 对准瓶身中心并稳定抓住瓶子。
Negative: 抓取偏向侧边，接触不稳，没能稳定抓住瓶子。
```

推理时只使用正向 guidance，例如：

```text
Align with the bottle center and secure a stable grasp.
```

中文示例：

```text
对准瓶身中心并稳定抓住瓶子。
```

`Progress / Stage / Issue / Action target` 这些变量仍然有用，但更适合作为上游中间语义，用来约束 text 生成，而不是直接原样暴露给 policy。

## 4. 训练方案

第一版默认保留 `value-like head`，因为它提供 reward grounding。

训练链条：

`reward / trajectory outcome -> value/progress-related estimator -> structured signals + current/future frames -> text supervision -> text-conditioned policy`

训练时可以利用未来帧，因此可以更稳地构造 text supervision。第一版建议：

- 输入当前帧 `o_t`
- 输入未来帧 `o_{t+Δ}`
- 输入 `value / progress` 相关信号
- 输入少量候选模板和 example
- 由 VLM 或标注器生成 1 到 2 句局部动作模式文本

这里推荐 `template-guided / example-guided generation`，而不是纯模板硬编码或完全自由生成。这样能同时利用：

- 当前帧和未来帧的视觉证据
- value / progress 的 reward grounding
- 模板和 example 的格式约束

训练目标仍然保持简单：

- `pi(a | o, task)`
- `pi(a | o, task, text_condition)`

保留原有 action loss，不改 diffusion / flow-matching 主训练目标，只在输入侧加入 text，并使用 condition dropout。

## 5. 推理方案

推理时不能使用未来帧，因此链条变成：

`current observation + value/progress-related signals -> positive guidance text -> policy`

关键点：

- 推理时只生成正向 guidance
- policy 在这个正向 guidance 下生成动作

目前可考虑三种方案：

- `方案 A：固定候选 guidance`
  - 最稳定，最适合第一版做可控实验
  - 但 guidance 粒度可能过粗

- `方案 B：额外的 VLM 生成 guidance`
  - guidance 生成和 action 生成解耦
  - 更容易单独评估 text quality
  - 但系统更复杂、推理链更长

- `方案 C：让 VLA 自己生成 guidance`
  - 模型统一
  - 但需要额外训练目标，且 text 和 action 容易一起漂移

第一版更推荐从方案 A 或方案 B 开始。

后续一个自然扩展是加入 `user prompt`，把 condition 写成：

`system guidance + user prompt -> action`

但这不建议放在第一版里一起做。

## 6. 第一版默认设置

- baseline：binary advantage-conditioned policy
- 上游：保留 `value / progress estimator`
- guidance 粒度：`action-chunk-level` 或 `local transition-level`
- 训练 supervision：用当前帧、未来帧和 progress 信号构造 `positive / negative` 文本
- 推理 condition：只生成正向 guidance
- text 生成：优先从 `固定候选 guidance` 或 `额外 VLM` 两种方案开始

## 7. 实现路线

### Step 1

固定 binary baseline，统一训练、推理和评估接口。

### Step 2

保持数据、estimator、loss 不变，用局部动作级别的 text supervision 替换 binary indicator；推理时只输入正向 guidance。

### Step 3

做训练侧 ablation：

- binary vs text
- structured short text vs structured tokens
- 不同 text 生成方式
- 不同 guidance 粒度

### Step 4

做推理侧验证：

- 固定候选 guidance vs 额外 VLM
- 不同正向 guidance 对动作输出的影响
- text 是否真的提升推理可控性

### Step 5

扩大到更多任务、更多 stage、更多 failure mode，验证它是否比 binary 更容易 scale。

## 8. 实验与数据

实验路线建议分三步：

1. 先做可控离线实验
2. 再做公开真实机器人数据上的验证
3. 最后做小规模自采在线数据验证

### 第一阶段：可控 benchmark

优先用容易快速迭代、能做局部动作分析的 benchmark：

- `CALVIN`
- `RLBench`
- `LIBERO`

这一阶段主要回答：

- binary 和 text 谁更好
- guidance 的合理粒度是什么
- 固定候选 guidance 和额外 VLM 哪个更稳

### 第三阶段：小规模自采在线数据

如果要验证推理时正向 guidance 是否真的改善 action generation，最终仍需要一定量的自采在线数据。

第一版可以先选：

- 瓶子/杯子抓取与放置
- 衣物展平或简单折叠

原因是这些任务更容易观察局部 positive / negative 动作模式，也更容易验证 guidance 是否真的改变动作。

## 9. 评估指标

除了 success rate，建议同时看：

- task completion time
- retry count
- recovery success rate
- dead-loop frequency
- 同一 observation 下，不同正向 guidance 是否会产生不同动作

## 10. 主要风险

- `text` 可能没有和真实 action quality 对齐，最后更像视觉描述而不是 policy improvement signal。
  验证方式：比较有无 value grounding，并检查 text 与后续成功率/return 的相关性。

- 训练能看未来帧、推理看不到，会带来 train-test mismatch。
  验证方式：比较 oracle text supervision 和 infer-time generated text。

- policy 可能并没有真正使用 text，而是主要依赖 observation。
  验证方式：固定 observation，替换 guidance，检查动作是否变化，并比较去掉 text condition 前后的性能。

- guidance 可能过粗、过噪，或者最后塌缩成少数固定句式。
  验证方式：比较不同 guidance 粒度，分析 guidance 多样性，以及 guidance 改变时动作是否真的改变。
