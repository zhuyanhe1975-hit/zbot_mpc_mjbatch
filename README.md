# ZBot MPC with mjbatch

使用 `mjbatch` 和采样式 MPC，在 MuJoCo 中实现 ZBot 的接触守卫原地交替踏步。模型资源复制自 `zbot_rl_unilab` 的 MuJoCo 场景；运行时不依赖该强化学习项目。`assets/provenance.json` 保留了模型来源和关节信息。

## 已实现

- 使用 `mjbatch` 对 13 维位置、12 维速度、6 个位置执行器的 ZBot 做并行 MuJoCo rollout；可读取身体与双脚位置、接触传感器，并限制关节位置目标的变化速度。
- 使用 `implicitfast` 积分器稳定运行原始站立位姿，并启用两次 MuJoCo NoSlip 后处理迭代以抑制软接触模型的缓慢滑动；六个关节采用与已验证 ZBot MuJoCo 模型一致的阻尼、摩擦和转子惯量，避免落地后持续振荡。
- 接触守卫的变频状态机：目标落地间隔为 `T = 1/f`。默认 0.5 Hz 的慢速模式每步先用约 0.9 秒转移重心，再用约 0.3 秒抬脚、0.5 秒保持、0.3 秒落脚。只有摆动脚连续至少 0.1 秒高于地面 3 mm、且最终双脚接触，才算有效换脚。若较快目标节拍没有安全候选，规划器可把当前一步延长，最长 2 秒。
- 参考经典 [Kajita 等人的 ZMP 预览控制](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)，先根据未来两步的支撑脚位置规划双脚支撑与单脚支撑的 ZMP 目标，再用线性倒立摆的发散分量倒推有界的质心参考。每次落地后，`mjbatch` 采样 MPC 从测量状态出发，优化独立的重心转移和抬脚关节脉冲，并跟踪质心与 CoP/ZMP 参考。预测覆盖一步和 0.5 秒站立恢复。
- MPC 在预测的 20 ms 控制节点上筛除支撑脚失去接触、摆动脚离地时支撑力低于 15 N、摆动脚没有持续抬起、或双脚间隙不足 2 mm 的候选。执行时再次检查支撑力。找不到候选时安全停止并输出 `terminated_early` 和 `stop_reason`，不会继续执行已知不安全的轨迹。
- 两脚碰撞网格的带符号距离用于防撞。MPC 淘汰预测控制节点中两脚距离不足 2 mm 或已接触的候选；实际仿真检测到两脚接触会立即停止。MuJoCo 网格距离偶尔返回零但最近点仍分离，代码用最近点线段修正这种情况，并始终以接触传感器复核。其他关节之间的自碰撞不在本次约束范围内。
- 全身重心来自 MuJoCo 的 `subtree_com`，足底支撑边界由站立时接近地面的碰撞网格顶点构造，并随足部姿态变换。每只脚的 CoP（平地单脚支撑时的 ZMP）由全部地面接触的合成位置、三维力和力矩计算。MPC 在实际单脚支撑的预测片段中加入重心投影与 CoP 越出足底的**软惩罚**，并输出最小支撑裕度；负值表示越界。
- `./run.sh` 默认运行 0.5 Hz、4.2 秒的慢速闭环踏步，并输出有效换脚次数、落地时刻、单脚支撑最小法向力、机身最低高度及最大水平漂移。站立基线仍可用 `uv run zbot-mpc-smoke` 单独验证。
- 支持 learned warm-start：策略根据步边界状态提出 18 维搜索中心，MPC 仍保留启发式种子和历史解，并对所有候选执行原有接触、支撑力、抬脚持续时间和脚间碰撞硬筛选。策略不能直接绕过安全约束。
- 长保持模式把 HOLD 分成五个控制结点，并在左右换脚之间执行 1 秒双脚支撑恢复。`--frequency 0.4 --hold-seconds 1.0` 会硬性要求每只脚连续高于地面 3 mm 至少 1 秒；左右脚使用独立的安全 warm-start。

当前控制器仍是**步级采样 MPC**，ZMP 预览是简化的线性倒立摆参考，关节执行由完整 MuJoCo 模型检验。硬筛选只检查离散预测节点，不能保证节点之间始终接触，也不能保证支撑脚完全静止；其位移仍是优化代价。质心和 CoP 的足底边界也是软代价。若规划找不到满足接触条件的下一步，会提前安全停止。当前六关节位置伺服模型缺少保证连续多步踏步所需的全身接触力控制；这是仿真原型，不应作为真实硬件控制器。

## 运行

```bash
uv sync --dev
./run.sh
./visualize.sh --viewer --frequency 0.5 --seconds 4.2 --threads 8
./visualize.sh --frequency 0.5 --seconds 4.2 --threads 8 --output dist/zbot_quasistatic.mp4
./run.sh --frequency 1 --seconds 10 --candidates 256 --iterations 16 --threads 8
uv run zbot-mpc-smoke
uv run pytest -q
```

左右脚依次抬起并保持至少 1 秒：

```bash
./visualize.sh --viewer --frequency 0.4 --hold-seconds 1.0 \
  --steps 2 --seconds 9 --candidates 64 --iterations 4 --threads 8
```

## 交互式回放

交互式 Viewer 按记录的实际闭环状态回放机器人，不叠加 MPC 预测轨迹。

```bash
./visualize.sh --viewer --frequency 0.5 --steps 2 --seconds 5 \
  --candidates 64 --iterations 4 --threads 8
```

## 0.5–2 Hz 周期踏步

周期模式会连续规划八次左右交替落脚，并自动选用已验证的搜索预算：0.5 Hz 使用 64 条候选、4 轮迭代，1–2 Hz 使用 128 条候选、8 轮迭代。每只脚上一条通过硬约束的计划会保留为下一周期的候选，但不会替代从实测状态重新优化。

```bash
uv run zbot-mpc-periodic --frequency 0.5 --steps 8 --threads 8
uv run zbot-mpc-periodic --frequency 1.0 --steps 8 --threads 8
uv run zbot-mpc-periodic --frequency 1.5 --steps 8 --threads 8
uv run zbot-mpc-periodic --frequency 2.0 --steps 8 --threads 8
```

交互回放时，0.5 Hz 使用：

```bash
./visualize.sh --viewer --frequency 0.5 --steps 8 --seconds 18 \
  --candidates 64 --iterations 4 --threads 8
```

1–2 Hz 使用：

```bash
./visualize.sh --viewer --frequency 2.0 --steps 8 --seconds 10 \
  --candidates 128 --iterations 8 --threads 8
```

结果中的 `achieved_landing_frequency_hz` 和 `cadence_tracking_ratio` 根据真实落脚时刻计算。安全候选不足时允许延长单步，因此实际频率可能低于指令，但当前回归要求平均落脚频率不低于指令的 75%。

## 单脚站立

单脚模式从双脚站立启动，先用 MPC 将质心/ZMP 移到指定支撑脚上方，再抬起另一只脚。随后依照参考脚本的质心雅可比修正，在保持质心投影位于支撑脚上方的同时，用默认 6 秒将六个关节平滑归零，最终展开为约 58.3 cm 高的立柱。运行时持续监测支撑丢失、支撑力、摔倒和双脚碰撞。

```bash
uv run zbot-mpc-single-leg --support-foot 0 --hold-seconds 5 --threads 8
uv run zbot-mpc-single-leg --support-foot 1 --hold-seconds 5 --threads 8
```

`--support-foot` 表示保持着地的脚（0 或 1）；`--unfold-seconds` 是从抬脚位姿展开到全关节归零的时间，`--hold-seconds` 表示成为立柱后的持续保持时间。

单腿模式使用 1 ms 物理步长、20 ms 控制周期，规划和执行采用相同的积分精度；默认搜索为 64 条候选、8 轮迭代。伸直采用两端速度和加速度为零的五次曲线，减少进入最终保持时的激励。摘要中的 `last_second_hold_qvel_rms` 和 `last_second_hold_base_motion_m` 分别记录最后最多 1 秒保持期间的广义速度 RMS 与机身各轴位置极差的范数，用于观察微抖；提前结束且未进入保持时为 `null`。

MuJoCo Viewer 中回放完整的双脚到单脚转换过程：

```bash
./single_leg_viewer.sh --support-foot 0 --hold-seconds 5 --threads 8
# 或：
uv run zbot-mpc-single-leg-viewer --support-foot 1 --hold-seconds 5 --threads 8
```

Viewer 中蓝色是支撑脚，橙色是抬起脚；空格暂停/继续，`N` 单帧前进，`R` 从头回放。`--speed 0.5` 可以慢放，`--once` 会在播放一遍后自动关闭。

## 逆向 MPC：单腿立柱回到双脚站立

逆向模式从正向单腿模式的实际最终状态继续：先折叠下降，再落脚并收稳，最后保持双脚站立。默认命令先在后台生成单腿立柱，回放只包含逆向过程；不会直接重置成双脚位姿。

正向控制序列的逆序仅作为搜索初值。`ReturnMPC` 从当前实测位置、速度和控制量进行正向物理 rollout，优化平滑的关节修正；初次规划覆盖整个回落及稳定阶段，折叠完成后再次用实测状态规划剩余落脚过程。候选必须保持原支撑脚接触，单支撑时法向力至少 15 N，双脚间隙至少 2 mm，机身高度至少 22 cm；末端要求两脚持续接触且各承载至少 1 N，并检查关节目标误差和残余速度。找不到安全候选时停止，不执行未经验证的参考序列。

```bash
uv run zbot-mpc-double-leg --support-foot 0 --hold-seconds 3 --threads 8
uv run zbot-mpc-double-leg --support-foot 1 --hold-seconds 3 --threads 8
# 回放单腿立柱到双脚站立：
./double_leg_viewer.sh --support-foot 0 --threads 8 --once
```

`--support-foot` 是起始单腿立柱的支撑脚。默认逆向搜索为 32 条候选、4 轮迭代，物理步长 1 ms、控制周期 20 ms；默认过渡约 8.98 秒，其中包含 1 秒落脚收稳，之后才开始 `--hold-seconds` 的保持。与现有步边界 MPC 一样，这属于离线阶段规划，不保证实时求解。

已有成功的 `SingleLegRun` 可通过 `simulate_double_leg_stand(support_foot=column.support_foot, source_run=column)` 直接继续其最终状态。摘要提供 `mpc_replans`、`safe_candidate_evaluations`、双脚连续承载时长和末端速度 RMS。Viewer 保留暂停、单帧、重播操作，不叠加 MPC 轨迹。

## MPC teacher 与 learned warm-start

第一阶段先用现有 MPC 产生安全专家计划，再拟合一个无额外运行时依赖的轻量策略：

```bash
uv run zbot-mpc-train-warmstart \
  --episodes 8 --steps 2 --seconds 6.0 --threads 8 \
  --output dist/zbot_warmstart.npz \
  --dataset dist/zbot_mpc_expert.npz

./run.sh --warm-start dist/zbot_warmstart.npz \
  --candidates 128 --iterations 8 --frequency 0.5 --steps 2 --seconds 6.0 --threads 8
```

观测去除了自由根的世界坐标 `x/y`，包含姿态、速度、关节位置目标、步态相位、接触、归一化法向力和相对支撑脚的质心位置。策略为两只摆动脚使用独立线性头并共享观测归一化，避免左右脚数据不平衡被单个线性映射平均。策略只决定采样分布的初始均值；输出会限制在 MPC 原有参数范围内，预测 rollout 的硬安全筛选仍是最终裁决者。

当前训练命令属于 MPC teacher bootstrap：第一轮做岭回归行为克隆，后续轮次让已学策略驱动 MPC，再聚合这些新状态上的 MPC 标签（DAgger）。目的是先验证 learned warm-start 能否在较少候选和迭代下保持安全成功率。下一阶段才在同一策略接口上加入受限 residual PPO；在 warm-start 基线稳定前，不允许 RL 直接修改执行器目标。

上面的 1 Hz 命令用于较快节拍的实验；无安全候选时会提前停止。优先检查默认慢速回放的支撑效果。

## 规划效率与质心雅可比

较快的实验配置可用 `./run.sh --frequency 0.5 --seconds 4.2 --candidates 128 --iterations 8 --threads 8`。本机同一种子下，它用约 4.8 秒完成 4.2 秒仿真及两次换脚，单脚支撑最小法向力 15.97 N；默认的 256 条候选、16 次迭代约需 18 秒，最小法向力 20.8 N。较少的候选降低了安全裕量，64 条候选、8 次迭代在初始状态找不到安全踏步。

运行 `uv run python -m zbot_mpc_mjbatch.jacobian_probe` 可复测 MuJoCo `mj_jacSubtreeCom` 的局部预测能力。在初始站立位姿附近，对 128 条关节脉冲，0.20 秒时质心相对支撑脚位移的 x/y 预测相关系数为 0.552/0.128；到 1.40 秒为 -0.193/-0.151。此时接触模式和自由根姿态已变化，站立位姿的运动学雅可比不能代替控制量到未来接触状态的导数。直接用它预筛 MPC 候选在实测中导致原本可完成的两步提前失败，因此当前控制器没有启用这种预筛。若继续做导数型 MPC，需在每个接触模式下沿名义轨迹计算 MuJoCo 的[状态转移导数](https://mujoco.readthedocs.io/en/latest/computation/#derivatives)，再用完整 rollout 复核接触切换。

`visualize.sh` 先用同一套 MPC 和状态机计算闭环轨迹，再通过 MuJoCo 渲染和 `ffmpeg` 导出 `dist/zbot_stepping.mp4`。脚本在 Linux 默认使用 EGL，在 macOS 默认使用 CGL；也可通过 `MUJOCO_GL` 指定后端。橙色为摆动脚，蓝色为支撑脚，地面显示接触点。可用 `--output result.mp4` 指定文件、`--speed 0.5` 慢放、`--width 1280 --height 720` 调整分辨率；宽高须为偶数。系统需安装 `ffmpeg`。

加 `--viewer` 可在 MuJoCo 交互窗口中循环回放同一条轨迹，保留实际关节位置、速度和控制量。空格暂停或继续、`N` 单帧前进、`R` 从头播放，关闭窗口退出；`--once` 播放一遍后自动关闭。Viewer 需要图形桌面；在 macOS 上脚本会自动使用 `mjpython` 启动。MP4 模式仍可用于无桌面环境。

`--frequency` 的有效范围是 1/3–2 Hz，表示目标节拍；`--hold-seconds` 是摆动脚连续高于地面 3 mm 的硬性时长要求。`maximum_planned_step_duration_s` 显示安全筛选要求的最长实际规划周期。`--candidates` 和 `--iterations` 控制优化工作量。若提前停止，`run.sh` 会输出已完成段的指标并以非零状态退出。站立 smoke 可传入 `--num-sims 64 --seconds 2 --threads 8`。

`run.sh` 的 `minimum_single_support_com_margin_m` 与 `minimum_single_support_cop_margin_m` 仅统计摆动脚离地且支撑脚接触的时刻；如果没有这样的时刻，显示 `null`。CoP 的法向力不足 1 N 时不定义，也不计入裕度。

本机参考运行（0.5 Hz、4.2 秒、256 条候选）完成 2 次有效换脚，单脚支撑时的最小法向力为 20.8 N，支撑脚失去接触和两脚碰撞均为 0 帧，最低机身高度为 0.2534 m，最大水平漂移为 0.0535 m。延长到更多步时若找不到安全候选，会提前停止。

抬脚与支撑的逐步指标包括：完成步骤中最低的抬脚峰值间隙 `minimum_step_peak_clearance_m`、连续高于 3 mm 的最短时长 `minimum_step_3mm_clearance_run_s`、保持阶段的最低支撑接触比例 `minimum_step_support_contact_fraction`、全阶段的 `minimum_step_support_contact_fraction_all_phases`、以及最大支撑脚水平位移 `maximum_step_support_xy_motion_m`。`minimum_single_support_force_n` 是摆动脚离地时支撑脚承受的最小法向力；`weak_single_support_frames` 统计低于 15 N 的控制帧。`support_contact_loss_frames` 和 `maximum_support_xy_motion_all_frames_m` 包含未完成的最后一步。`minimum_foot_foot_distance_m` 与 `foot_foot_contact_frames` 检查两脚防撞。

依赖 `mjbatch` 通过 `../../AI/mjbatch` 的本地源码安装；若项目被移至其他机器，需相应调整 `pyproject.toml` 的路径来源，或改从包索引安装。

## 后续验证

1. 对 0.5–2 Hz 做受扰动和不同初始状态的验证，尤其检查 2 Hz 的落地延迟与短暂离地、低频长时间保持，以及 10 秒内约 0.1 m 级的水平漂移。
2. 加入步内短视野反馈和机身姿态约束，进一步降低受扰动时的漂移。
3. 在 Motrix 中复测，识别两种物理后端的接触差异。

本机初测：128 条候选、100 个控制节点（2 秒）、每节点 4 个 MuJoCo 子步，仅物理推进约需 101 毫秒，超过 20 毫秒控制周期；因此当前 MPC 在步边界做规划，并不满足 20 毫秒实时控制周期。
