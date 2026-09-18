# ZBot MPC with mjbatch

使用 `mjbatch` 和采样式 MPC，在 MuJoCo 中实现 ZBot 的接触守卫原地交替踏步。模型资源复制自 `zbot_rl_unilab` 的 MuJoCo 场景；运行时不依赖该强化学习项目。`assets/provenance.json` 保留了模型来源和关节信息。

## 已实现

- 使用 `mjbatch` 对 13 维位置、12 维速度、6 个位置执行器的 ZBot 做并行 MuJoCo rollout；可读取身体与双脚位置、接触传感器，并限制关节位置目标的变化速度。
- 使用 `implicitfast` 积分器稳定运行原始站立位姿，并启用两次 MuJoCo NoSlip 后处理迭代以抑制软接触模型的缓慢滑动。
- 接触守卫的变频状态机：目标落地间隔为 `T = 1/f`。默认 0.5 Hz 的慢速模式每步先用约 0.9 秒转移重心，再用约 0.3 秒抬脚、0.5 秒保持、0.3 秒落脚。只有摆动脚连续至少 0.1 秒高于地面 3 mm、且最终双脚接触，才算有效换脚。若较快目标节拍没有安全候选，规划器可把当前一步延长，最长 2 秒。
- 参考经典 [Kajita 等人的 ZMP 预览控制](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)，先根据未来两步的支撑脚位置规划双脚支撑与单脚支撑的 ZMP 目标，再用线性倒立摆的发散分量倒推有界的质心参考。每次落地后，`mjbatch` 采样 MPC 从测量状态出发，优化独立的重心转移和抬脚关节脉冲，并跟踪质心与 CoP/ZMP 参考。预测覆盖一步和 0.5 秒站立恢复。
- MPC 在预测的 20 ms 控制节点上筛除支撑脚失去接触、摆动脚离地时支撑力低于 15 N、摆动脚没有持续抬起、或双脚间隙不足 2 mm 的候选。执行时再次检查支撑力。找不到候选时安全停止并输出 `terminated_early` 和 `stop_reason`，不会继续执行已知不安全的轨迹。
- 两脚碰撞网格的带符号距离用于防撞。MPC 淘汰预测控制节点中两脚距离不足 2 mm 或已接触的候选；实际仿真检测到两脚接触会立即停止。MuJoCo 网格距离偶尔返回零但最近点仍分离，代码用最近点线段修正这种情况，并始终以接触传感器复核。其他关节之间的自碰撞不在本次约束范围内。
- 全身重心来自 MuJoCo 的 `subtree_com`，足底支撑边界由站立时接近地面的碰撞网格顶点构造，并随足部姿态变换。每只脚的 CoP（平地单脚支撑时的 ZMP）由全部地面接触的合成位置、三维力和力矩计算。MPC 在实际单脚支撑的预测片段中加入重心投影与 CoP 越出足底的**软惩罚**，并输出最小支撑裕度；负值表示越界。
- `./run.sh` 默认运行 0.5 Hz、4.2 秒的慢速闭环踏步，并输出有效换脚次数、落地时刻、单脚支撑最小法向力、机身最低高度及最大水平漂移。站立基线仍可用 `uv run zbot-mpc-smoke` 单独验证。

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

上面的 1 Hz 命令用于较快节拍的实验；无安全候选时会提前停止。优先检查默认慢速回放的支撑效果。

## 规划效率与质心雅可比

较快的实验配置可用 `./run.sh --frequency 0.5 --seconds 4.2 --candidates 128 --iterations 8 --threads 8`。本机同一种子下，它用约 4.8 秒完成 4.2 秒仿真及两次换脚，单脚支撑最小法向力 15.97 N；默认的 256 条候选、16 次迭代约需 18 秒，最小法向力 20.8 N。较少的候选降低了安全裕量，64 条候选、8 次迭代在初始状态找不到安全踏步。

运行 `uv run python -m zbot_mpc_mjbatch.jacobian_probe` 可复测 MuJoCo `mj_jacSubtreeCom` 的局部预测能力。在初始站立位姿附近，对 128 条关节脉冲，0.20 秒时质心相对支撑脚位移的 x/y 预测相关系数为 0.552/0.128；到 1.40 秒为 -0.193/-0.151。此时接触模式和自由根姿态已变化，站立位姿的运动学雅可比不能代替控制量到未来接触状态的导数。直接用它预筛 MPC 候选在实测中导致原本可完成的两步提前失败，因此当前控制器没有启用这种预筛。若继续做导数型 MPC，需在每个接触模式下沿名义轨迹计算 MuJoCo 的[状态转移导数](https://mujoco.readthedocs.io/en/latest/computation/#derivatives)，再用完整 rollout 复核接触切换。

`visualize.sh` 先用同一套 MPC 和状态机计算闭环轨迹，再通过 MuJoCo 渲染和 `ffmpeg` 导出 `dist/zbot_stepping.mp4`。脚本在 Linux 默认使用 EGL，在 macOS 默认使用 CGL；也可通过 `MUJOCO_GL` 指定后端。橙色为摆动脚，蓝色为支撑脚，地面显示接触点。可用 `--output result.mp4` 指定文件、`--speed 0.5` 慢放、`--width 1280 --height 720` 调整分辨率；宽高须为偶数。系统需安装 `ffmpeg`。

加 `--viewer` 可在 MuJoCo 交互窗口中循环回放同一条轨迹，保留实际关节位置、速度和控制量。空格暂停或继续、`N` 单帧前进、`R` 从头播放，关闭窗口退出；`--once` 播放一遍后自动关闭。Viewer 需要图形桌面；在 macOS 上脚本会自动使用 `mjpython` 启动。MP4 模式仍可用于无桌面环境。

`--frequency` 的有效范围是 0.5–2 Hz，表示目标节拍；`maximum_planned_step_duration_s` 显示安全筛选要求的最长实际规划周期。`--candidates` 和 `--iterations` 控制优化工作量。若提前停止，`run.sh` 会输出已完成段的指标并以非零状态退出。站立 smoke 可传入 `--num-sims 64 --seconds 2 --threads 8`。

`run.sh` 的 `minimum_single_support_com_margin_m` 与 `minimum_single_support_cop_margin_m` 仅统计摆动脚离地且支撑脚接触的时刻；如果没有这样的时刻，显示 `null`。CoP 的法向力不足 1 N 时不定义，也不计入裕度。

本机参考运行（0.5 Hz、4.2 秒、256 条候选）完成 2 次有效换脚，单脚支撑时的最小法向力为 20.8 N，支撑脚失去接触和两脚碰撞均为 0 帧，最低机身高度为 0.2534 m，最大水平漂移为 0.0535 m。延长到更多步时若找不到安全候选，会提前停止。

抬脚与支撑的逐步指标包括：完成步骤中最低的抬脚峰值间隙 `minimum_step_peak_clearance_m`、连续高于 3 mm 的最短时长 `minimum_step_3mm_clearance_run_s`、保持阶段的最低支撑接触比例 `minimum_step_support_contact_fraction`、全阶段的 `minimum_step_support_contact_fraction_all_phases`、以及最大支撑脚水平位移 `maximum_step_support_xy_motion_m`。`minimum_single_support_force_n` 是摆动脚离地时支撑脚承受的最小法向力；`weak_single_support_frames` 统计低于 15 N 的控制帧。`support_contact_loss_frames` 和 `maximum_support_xy_motion_all_frames_m` 包含未完成的最后一步。`minimum_foot_foot_distance_m` 与 `foot_foot_contact_frames` 检查两脚防撞。

依赖 `mjbatch` 通过 `../../AI/mjbatch` 的本地源码安装；若项目被移至其他机器，需相应调整 `pyproject.toml` 的路径来源，或改从包索引安装。

## 后续验证

1. 对 0.5–2 Hz 做受扰动和不同初始状态的验证，尤其检查 2 Hz 的落地延迟与短暂离地、低频长时间保持，以及 10 秒内约 0.1 m 级的水平漂移。
2. 加入步内短视野反馈和机身姿态约束，进一步降低受扰动时的漂移。
3. 在 Motrix 中复测，识别两种物理后端的接触差异。

本机初测：128 条候选、100 个控制节点（2 秒）、每节点 4 个 MuJoCo 子步，仅物理推进约需 101 毫秒，超过 20 毫秒控制周期；因此当前 MPC 在步边界做规划，并不满足 20 毫秒实时控制周期。
