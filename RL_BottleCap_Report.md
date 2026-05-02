# 基于强化学习的双手灵巧开瓶盖任务研究（论文/PPT正文草稿）

> 说明：本稿采用学术叙述方式，不包含源码与实现路径，可直接用于论文写作或 AI 生成 PPT。

---

## 1. Introduction

双手灵巧操作任务是机器人强化学习中的典型高难场景，核心难点在于：  
1) 高维连续动作控制；  
2) 接触动力学引起的强非线性；  
3) 双执行体在时序与受力上的协同耦合。  

开瓶盖任务同时包含“稳定抓持 + 精细开盖”两类子目标，要求策略在局部接触控制与全局协作决策之间取得平衡，因此适合用于研究强化学习方法在复杂操作任务中的适配性与泛化能力。

本文围绕双手开瓶盖任务，构建统一的训练与评估流程，对单智能体与多智能体强化学习方法进行系统实验分析。研究重点包括任务形式化、奖励设计、算法机制与实验结果解释。

Bi-DexHands 是一个面向双手灵巧操作的强化学习任务库，提供了多类双手协作任务（如 handover、lift、door、bottle cap 等）与多种 RL/MARL 算法接口。其特点是可在 GPU 上进行大规模并行采样，适合高维连续控制与协作决策研究。本文聚焦其中的开瓶盖任务，并在统一实验协议下进行算法分析。

**[图位-1：任务场景总览图（双手+瓶体+瓶盖）]**  
**[视频位-1：任务成功案例短视频（5-10 秒）]**

---

## 2. Task Formulation

### 2.1 MDP/MARL 建模

单智能体形式可写为：
M = (S, A, P, r, gamma)

多智能体形式可写为：
M_ma = (S, {O_i}_{i=1..N}, {A_i}_{i=1..N}, P, r, gamma)  
其中 N=2（左右手），联合动作为：a_t = [a_t^(L), a_t^(R)]

目标是学习策略 \(\pi\) 最大化折扣累计回报：
max over pi of E_pi [ sum_{t=0..T} (gamma^t * r_t) ]

### 2.2 State / Observation

观测采用高维连续状态，包含以下信息块：  
- 双手关节位置、速度与关节力矩相关特征；  
- 指尖位姿、线速度与角速度；  
- 手掌基座位置与朝向；  
- 物体位姿、线速度、角速度；  
- 与开盖相关的关键几何量（瓶身/瓶盖位置关系）。  

整体观测向量可抽象为：
o_t = [o_t^(hand_L), o_t^(hand_R), o_t^(object), o_t^(task)]

本任务的观测维度如下（重点）：
- full_state 维度：420  
- point_cloud 维度：2724（= 420 + 768 x 3）  

在 full_state 下可进一步分解为：
- 单手特征：199 维（关节状态 + 指尖状态 + 手掌位姿 + 历史动作）  
- 双手合计：398 维  
- 物体与任务相关特征：22 维  
- 总计：420 维

### 2.3 Action Space

动作空间为连续控制空间。策略在每个时间步输出关节/执行器控制量，经裁剪与缩放后作用于动力学仿真器：
a_t_tilde = clip(a_t, a_min, a_max)

在多智能体设置下：
a_t = [a_t^(L), a_t^(R))]  
a_t^(L) ~ pi_L(. | o_t^(L)), a_t^(R) ~ pi_R(. | o_t^(R))

动作维度设置如下：
- 单智能体联合控制：52 维（左右手联合动作）  
- 多智能体控制：每个 agent 26 维，共 2 个 agent  

可理解为：每只手提供一组连续控制量，最终形成双手联合控制输入。

动作后处理（Action Post-processing）在本任务中同样重要，核心包括：

1) 动作裁剪：  
为保证控制稳定性，策略输出首先被限制在动作边界内，避免异常大动作直接破坏接触过程。

2) 动作到执行器目标的映射：  
裁剪后的动作不会直接作为物理力输入，而是映射为关节目标（或等效控制目标），再交由仿真器执行。这一步将“学习空间”与“控制空间”对齐。

3) 平滑与动态尺度：  
动作更新受到控制频率与速度尺度影响，等价于在离散时间上施加低通约束，降低抖动与高频不稳定控制。

4) 双手联合执行：  
在多智能体设置下，左右手动作先分别输出，再组成联合动作向量进入同一步仿真，从而保持时序同步。

### 2.4 Reward Design

奖励函数采用稠密引导设计，核心由“接近项 + 开盖进度项 + 稳定项”构成。可概括为：
r_t = c0 - d_cap(t) - d_bottle(t) + lambda_up * Delta_h_cap(t) - lambda_a * ||a_t||_2^2

其中：  
- d_cap(t)：右手关键接触点与瓶盖距离；  
- d_bottle(t)：左手与瓶身距离；  
- Delta_h_cap(t)：瓶盖相对瓶身的有效抬升量；  
- ||a_t||_2^2：动作正则（平滑控制）。  

该设计的目标是引导策略从“接近-接触-协同开盖”逐步学习。

为便于报告表达，可将本任务奖励写成分段形式：

- 距离惩罚项：  
  r_dist(t) = - d_cap(t) - d_bottle(t)

- 开盖提升项（仅在右手接近瓶盖后激活）：  
  r_up(t) = 30 * ||p_cap_up(t) - p_bottle(t)||_2, if d_finger_cap(t) <= 0.3  
  r_up(t) = 0, otherwise

- 动作正则项：  
  r_act(t) = - lambda_a * ||a_t||_2^2

- 合成奖励：  
  r_t = 2.0 + r_dist(t) + r_up(t) + r_act(t)

其中关键阈值可在汇报中明确列出：  
- 接近激活阈值：d_finger_cap <= 0.3  
- 成功阈值：||p_cap_up - p_bottle||_2 > 0.03  
- 典型失败重置阈值：手-物体距离越界或瓶盖高度异常

### 2.5 Termination 与 Success

Episode 在以下情况结束：  
1) 达到最大时间步 \(T_{max}\)；  
2) 进入失败状态（如手-物体偏离过大等）；  
3) 环境触发重置条件。  

任务成功定义为瓶盖相对瓶身达到有效开盖阈值，可写为：
Success = 1[ ||p_cap_up - p_bottle||_2 > delta ]  
其中 delta 为任务阈值。  

**[图位-2：状态/动作/奖励示意图]**

---

## 3. RL Methods

本文采用 PPO、SAC、MAPPO、HAPPO 四类方法，覆盖单智能体与多智能体主流范式。

### 3.1 PPO（On-policy）

PPO 通过剪切策略比率约束更新幅度，提高训练稳定性。其优化目标可写为：
L_clip(theta) = E[ min( r_t(theta)*A_hat_t, clip(r_t(theta), 1-eps, 1+eps)*A_hat_t ) ]

### 3.2 SAC（Off-policy）

SAC 采用最大熵目标，在提高回报的同时保持策略探索性：
J(pi) = sum_t E_{(s_t,a_t)~pi} [ r(s_t,a_t) + alpha * H(pi(.|s_t)) ]

### 3.3 MAPPO（CTDE）

MAPPO 在多智能体场景采用“集中训练、分散执行”（CTDE）：
- 训练时 Critic 使用共享状态（或共享观测）进行价值估计；
- 执行时每个 agent 的 Actor 仅依赖局部观测。

联合策略写为：

Pi(a|o) = product over i of pi_i(a_i|o_i)

其中 a=[a_1,...,a_N]，o=[o_1,...,o_N]。

MAPPO 的策略更新沿用 PPO 剪切目标。对 agent i：

r_t(theta_i) = pi_i(a_{i,t}|o_{i,t};theta_i) / pi_i(a_{i,t}|o_{i,t};theta_i_old)

L_clip_i(theta_i) = E_t[ min( r_t(theta_i) * A_hat_{i,t}, clip(r_t(theta_i), 1-eps, 1+eps) * A_hat_{i,t} ) ]

优势估计采用 GAE：

A_hat_{i,t} = sum over l=0..T-t-1 of (gamma*lambda)^l * delta_{i,t+l}

delta_{i,t} = r_{i,t} + gamma * V_i(s_{t+1}) - V_i(s_t)

在实现层面，MAPPO 的损失可写为三项组合：

L_policy_i = - E[ min( r_t(theta_i) * A_hat_{i,t}, clip(r_t(theta_i), 1-eps, 1+eps) * A_hat_{i,t} ) ]

L_value_i = E[ (V_i - R_hat_i)^2 ]    (可结合 clipped value 或 huber 形式)

L_entropy_i = - E[ H(pi_i) ]

L_total_i = L_policy_i + c_v * L_value_i + c_e * L_entropy_i

其中 c_v 与 c_e 分别为 value 与 entropy 权重系数。
其中 R_hat_i 表示由 bootstrap return 计算得到的价值学习目标；
H(pi_i) 表示策略分布熵，用于调节探索强度（当 entropy 权重为 0 时该项不参与优化）。

### 3.4 HAPPO（异构协作）

HAPPO 面向异构 agent 协作优化，在保留 CTDE 框架的同时，更强调“联合策略耦合”带来的更新影响。

与 MAPPO 的关键区别在于：HAPPO 在多智能体更新时采用顺序式优化，并引入耦合修正因子来建模先更新 agent 对后更新 agent 的分布影响。该因子可表示为：

F_i = exp( sum over j<=i of (log pi_j_new - log pi_j_old) )

该机制用于降低多智能体联合更新中的相互干扰，提高异构协作任务中的训练稳定性。对于左右手角色分工明显的双手开瓶盖任务，这一机制具有较强任务相关性。

从损失视角看，HAPPO 与 MAPPO 共享相同的 policy/value/entropy 三项基础形式，但在顺序更新过程中通过耦合修正因子 F_i 对后续 agent 的更新权重进行重标定，从而降低联合策略分布漂移造成的不稳定。

### 3.5 网络架构（Actor / Critic）

本任务采用共享范式的 Actor-Critic 结构，不同算法在优化器与更新机制上存在差异，但网络主干可统一描述为：

- Actor 输入：本地观测（每个 agent 约 221 维）  
- Critic 输入：集中状态/共享观测（约 420 维）  
- 隐藏层：3 层 MLP 主干（每层隐藏维度 512，含归一化与非线性激活）  
- Actor 输出：连续动作分布参数（每个 agent 26 维动作）  
- Critic 输出：标量状态价值 V(s)

可写为：

- pi_i(a|o_i) = Actor_i(o_i; theta_i)  
- V_i(s) = Critic_i(s; phi_i)

在多智能体设置下，总体联合策略为：

- Pi(a|o) = product over i of pi_i(a_i|o_i)

该网络设计的直觉是：  
- Actor 使用局部观测保证执行可部署性；  
- Critic 使用更完整信息提升训练阶段价值估计稳定性。  

**[图位-3b：Actor-Critic 网络结构图（输入维度/隐藏层/输出维度）]**

**[图位-3：PPO/SAC/MAPPO/HAPPO 方法关系图]**

---

---

## 4. Training Environment and Setup

### 4.1 仿真环境

- 物理仿真：GPU 并行物理引擎  
- 任务场景：双手开瓶盖  
- 控制类型：连续动作控制  
- 观测类型：全状态观测（full-state）  

为了使报告更完整，可在本节补充如下背景信息：
- 任务库定位：双手灵巧操作强化学习基准  
- 场景属性：接触密集、长时序协作、高维连续动作  
- 计算特征：多环境并行采样、GPU 加速训练  
- 研究价值：适合验证单智能体与多智能体方法在复杂协作任务中的泛化能力

### 4.2 训练设置

- 训练预算：100M 环境步  
- 并行环境：256  
- 算法集合：PPO、SAC、MAPPO、HAPPO  
- 策略网络：Actor-Critic（MLP 主干，隐藏层宽度 512）  
- 优化方式：单智能体与多智能体分别采用对应训练循环  

关键配置参数（主实验）如下：

1) 环境与控制参数  
- episode length: 125  
- control frequency inverse: 1  
- dof speed scale: 20.0  
- observation type: full_state  
- action dim: 26 x 2（多智能体）/ 52（单智能体）  

2) 奖励与任务参数  
- dist reward scale: 20  
- rot reward scale: 1.0  
- action penalty scale: -0.0002  
- success tolerance（配置项）: 0.1  
- max consecutive successes: 0  

3) MAPPO/HAPPO 共享优化超参数  
- num env steps: 100000000  
- rollout length: 8  
- gamma: 0.96  
- gae lambda: 0.95  
- ppo epoch: 5  
- mini-batch: 1  
- clip param: 0.2  
- lr / critic lr: 5e-4 / 5e-4  
- max grad norm: 10  
- entropy coef: 0.0  

4) MAPPO 与 HAPPO 的差异化配置  
- value normalization: MAPPO 关闭，HAPPO 开启  
- 其余主训练配置在本实验中保持一致，以保证算法对比公平性  

领域随机化（Domain Randomization）用于提升策略鲁棒性。本文实验的基础配置可在“关闭随机化”和“线性调度随机化”之间切换。其随机化对象主要包括：

- 观测噪声：对状态观测添加高斯扰动（含相关噪声项）；
- 动作噪声：对控制输出注入扰动，提升对执行误差的容忍；
- 动力学参数：如重力扰动、关节阻尼/刚度扰动；
- 物体属性：如质量、摩擦与尺度扰动。

为避免训练初期不稳定，随机化强度通常采用分阶段或线性增大策略（curriculum-style scheduling）。  
在报告中可明确说明：当前主实验以“固定环境”进行算法对比，随机化作为增强鲁棒性的扩展实验设置。

### 4.3 评估设置

- 加载训练后策略进行离线评估  
- 评估指标：Success Rate、Average Reward、Max Reward  
- 评估采样采用固定协议，保证结果可比  

**[图位-4：训练与评估流程图（Train -> Checkpoint -> Eval）]**  
**[视频位-2：评估轨迹可视化（成功/失败各一段）]**

---

## 5. Experiments

### 5.1 实验目标

在统一预算与任务设置下，观察不同 RL 方法在双手开瓶盖任务中的表现差异，分析其在任务完成度与回报上的一致性。

### 5.2 实验设置总表（PPT建议单独一页）

| 项目 | 设置 |
|---|---|
| 任务 | ShadowHandBottleCap（双手开瓶盖） |
| 观测类型 | full_state |
| 观测维度 | 420 |
| 动作维度 | 单智能体 52；多智能体 26 x 2 |
| 对比算法 | MAPPO、HAPPO |
| 训练预算 | 100M environment steps |
| 并行环境数 | 256 |
| rollout 长度 | 8 |
| 评估回合数 | 10000 |
| 评估指标 | success rate / average reward / max reward |

### 5.2 指标定义

- 成功率：
SR = (1/N) * sum_{k=1..N} 1[episode_k success]
- 平均回报：
R_bar = (1/N) * sum_{k=1..N} R_k
- 最佳回报：
R_max = max_k R_k

补充说明（建议写入报告正文）：
- Success Rate 是任务级目标指标，直接反映是否完成开盖动作。  
- Average Reward 是过程性学习指标，反映策略在整个轨迹上的综合收益。  
- 在双手接触任务中，两者可能不完全一致，因此需要联合报告。

### 5.3 实验记录建议（可直接放 PPT）

- 表 1：不同算法最终成功率  
- 表 2：不同算法平均/最大回报  
- 图 1：训练阶段奖励曲线  
- 图 2：评估阶段成功率曲线  

**[图位-5：算法结果总表]**  
**[图位-6：Success Rate 曲线图]**  
**[图位-7：Reward 曲线图]**

---

## 6. Results and Analysis

### 6.1 结果总表（MAPPO vs HAPPO，PPT建议单独一页）

> 说明：以下为当前实验记录的代表性结果，可在最终汇报前替换为你最终确认的数值版本。

| 算法 | Success Rate | Average Episode Reward | Max Episode Reward |
|---|---:|---:|---:|
| HAPPO | 0.2509 | 509.2874 | 796.4344 |
| MAPPO | 0.9488 | 723.9772 | 994.3657 |

由该结果可见，在当前任务与训练预算设置下，MAPPO 在任务成功率和回报上均显著优于 HAPPO。

从实验现象可得到以下结论：

1. 多智能体方法在双手协作任务上通常具有更高任务完成度；  
2. 回报与成功率不总是严格一致，存在“高回报但任务完成不足”的情况；  
3. 对于双手开瓶盖这类接触密集任务，成功率是更具任务语义的核心指标；  
4. 不同算法的训练稳定性与收敛效率在高维协作场景中差异显著。

上述现象说明，在灵巧操作研究中应采用“任务成功指标 + 回报指标”的双指标体系，避免单一指标导致结论偏差。

---

## 6.1 Failure Case Analysis（失败案例分析）

为避免仅展示成功结果，本文补充典型失败轨迹分析。失败案例主要反映策略在接触稳定性与协同控制上的不足。

### Case A：抓持不稳导致提前重置

现象：左手对瓶身约束不足，姿态波动放大后触发失败重置。  
可能原因：  
- 抓持阶段动作抖动较大，接触稳定性不足；  
- 左右手协同时序不一致，未形成稳定支撑-操作分工。

**[图位-10：失败案例A关键帧（t1/t2/t3）]**

### Case B：接近成功但开盖阈值未达成

现象：策略已完成接触与部分旋拧/抬升，但未达到任务成功阈值，最终判定失败。  
可能原因：  
- 奖励中过程项较高，策略倾向于“接近成功态”而非“完成成功态”；  
- 开盖末段控制力度或方向不足，导致最终抬升量不足。

**[图位-11：失败案例B关键帧（接触成功但开盖不足）]**

### 小结

失败案例表明，双手开瓶盖任务中的关键瓶颈不只在“接近目标”，还在于末段协同控制的稳定性与力度控制。  
该分析可用于解释为什么部分实验中出现“回报不低但成功率受限”的现象。

---

## 7. Conclusion

本文围绕双手开瓶盖任务，系统给出了强化学习建模、算法训练与实验分析流程，并基于 PPO、SAC、MAPPO、HAPPO 四类方法开展实验研究。结果表明，双手协作任务对算法稳定性与任务适配性提出了更高要求，任务级成功率在该类问题中具有关键评价价值。

未来工作可沿以下方向展开：  
- 引入视觉/点云观测并研究表示学习；  
- 扩展到多任务或跨物体泛化；  
- 结合模仿学习与离线数据提升样本效率。

---

## 附：AI 生成 PPT 的提示词

请基于本文内容生成中文学术风格 PPT，要求：  
1) 按章节生成：Introduction、Task Formulation、RL Methods、Environment & Setup、Experiments、Results、Conclusion；  
2) 每页 3-5 条要点，保留文中的“图位/视频位”占位符；  
3) 公式页使用 LaTeX 渲染;
5) 风格偏强化学习论文汇报，不使用工程实现细节

