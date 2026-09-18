# ZBot MPC with mjbatch

使用 `mjbatch` 和采样式 MPC，在 MuJoCo 中实现 ZBot 的接触守卫原地交替踏步。模型资源复制自 `zbot_rl_unilab` 的 MuJoCo 场景；运行时不依赖该强化学习项目。`assets/provenance.json` 保留了模型来源和关节信息。

## 已实现

- 使用 `mjbatch` 对 13 维位置、12 维速度、6 个位置执行器的 ZBot 做并行 MuJoCo rollout；可读取身体与双脚位置、接触传感器，并限制关节位置目标的变化速度。
- 使用 `implicitfast` 积分器稳定运行原始站立位姿。当前场景的执行器参数在 MuJoCo 默认 Euler 积分器下数步内失稳，因此必须在构造 `Batch` 前选定积分器。
- 接触守卫的变频状态机：相邻交替落地的目标间隔为 `T = 1/f`，依次经过重心转移、抬脚、单脚保持、落脚、等待接触；只有目标脚曾离地且双脚重新接触才换侧，原地站着不会被算作一次踏步。改变频率时，在下一次有效换侧后生效。固定阶段时长合计 0.32 秒，剩余时间留给单脚保持，因此 0.5 Hz 保持 1.68 秒，2 Hz 保持 0.18 秒。
- 每次有效落地后，MPC 从当前测量的关节状态、速度和执行器目标出发，用 `mjbatch` 批量评估下一步的关节目标脉冲。快速节拍增加一个短暂的重心转移脉冲。预测覆盖完整一步和 0.5 秒站立恢复；代价约束真实足底离地间隙、支撑脚接触与法向力、支撑脚水平移动、机身高度、全身重心漂移、躯干姿态及终端速度。只执行当前一步，下次落地重新优化。
- 两脚碰撞网格的带符号距离用于防撞。MPC 淘汰预测控制节点中两脚距离不足 2 mm 或已接触的候选；实际仿真检测到两脚接触会立即停止。MuJoCo 网格距离偶尔返回零但最近点仍分离，代码用最近点线段修正这种情况，并始终以接触传感器复核。其他关节之间的自碰撞不在本次约束范围内。
- 全身重心来自 MuJoCo 的 `subtree_com`，足底支撑边界由站立时接近地面的碰撞网格顶点构造，并随足部姿态变换。每只脚的 CoP（平地单脚支撑时的 ZMP）由全部地面接触的合成位置、三维力和力矩计算。MPC 在实际单脚支撑的预测片段中加入重心投影与 CoP 越出足底的**软惩罚**，并输出最小支撑裕度；负值表示越界。
- `./run.sh` 默认运行 1 Hz、5 秒的闭环踏步，并输出有效换脚次数、落地时刻、机身最低高度及最大水平漂移。站立基线仍可用 `uv run zbot-mpc-smoke` 单独验证。

当前控制器是**步级采样 MPC**：每步落地时重规划，步内按优化轨迹执行并由接触守卫决定换脚。重心和 CoP 项是软代价，不能保证每个时刻都在支撑边界内；动态踏步时重心投影短暂越界也不等于失稳。足底边界按站立姿态的网格近地顶点近似，倾斜接触时有几何误差。它仍是 MuJoCo 仿真原型，尚未加入步内高频平衡修正或真实硬件控制。

## 运行

```bash
uv sync --dev
./run.sh
./run.sh --frequency 1 --seconds 10 --candidates 256 --iterations 16 --threads 8
./visualize.sh --frequency 1 --seconds 5
./visualize.sh --viewer --frequency 1 --seconds 5
uv run zbot-mpc-smoke
uv run pytest -q
```

`visualize.sh` 先用同一套 MPC 和状态机计算闭环轨迹，再通过 MuJoCo 的 EGL 渲染和 `ffmpeg` 导出 `dist/zbot_stepping.mp4`，无需图形桌面。橙色为摆动脚，蓝色为支撑脚，地面显示接触点。可用 `--output result.mp4` 指定文件、`--speed 0.5` 慢放、`--width 1280 --height 720` 调整分辨率；宽高须为偶数。系统需安装 `ffmpeg`。

加 `--viewer` 可在 MuJoCo 交互窗口中循环回放同一条轨迹，保留实际关节位置、速度和控制量。空格暂停或继续、`N` 单帧前进、`R` 从头播放，关闭窗口退出；`--once` 播放一遍后自动关闭。Viewer 需要图形桌面，MP4 模式仍可用于无桌面环境。

`--frequency` 的有效范围是 0.5–2 Hz。`--candidates` 和 `--iterations` 控制优化工作量。本机默认参数下，0.5、1、2 Hz 各自通过了 10 秒的 MuJoCo 仿真；这仍不是受扰动或真实硬件的稳定性保证。站立 smoke 可传入 `--num-sims 64 --seconds 2 --threads 8`。

`run.sh` 的 `minimum_single_support_com_margin_m` 与 `minimum_single_support_cop_margin_m` 仅统计摆动脚离地且支撑脚接触的时刻；如果没有这样的时刻，显示 `null`。CoP 的法向力不足 1 N 时不定义，也不计入裕度。

抬脚与支撑的逐步指标包括：完成步骤中最低的抬脚峰值间隙 `minimum_step_peak_clearance_m`、连续高于 3 mm 的最短时长 `minimum_step_3mm_clearance_run_s`、保持阶段的最低支撑接触比例 `minimum_step_support_contact_fraction`、最大支撑脚水平位移 `maximum_step_support_xy_motion_m`。`minimum_foot_foot_distance_m` 与 `foot_foot_contact_frames` 用于检查两脚防撞。MPC 的硬筛选在 20 ms 控制节点上进行，不等同于连续时间的形式化防撞保证；若找不到安全候选，规划会停止。

依赖 `mjbatch` 通过 `../../AI/mjbatch` 的本地源码安装；若项目被移至其他机器，需相应调整 `pyproject.toml` 的路径来源，或改从包索引安装。

## 后续验证

1. 对 0.5–2 Hz 做受扰动和不同初始状态的验证，尤其检查 2 Hz 的落地延迟与短暂离地、低频长时间保持，以及 10 秒内约 0.1 m 级的水平漂移。
2. 加入步内短视野反馈和机身姿态约束，进一步降低受扰动时的漂移。
3. 在 Motrix 中复测，识别两种物理后端的接触差异。

本机初测：128 条候选、100 个控制节点（2 秒）、每节点 4 个 MuJoCo 子步，仅物理推进约需 101 毫秒，超过 20 毫秒控制周期；因此当前 MPC 在步边界做规划，并不满足 20 毫秒实时控制周期。
