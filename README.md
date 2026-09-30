# ZBot MPC with mjbatch

使用 `mjbatch` 和采样式 MPC，在 MuJoCo 中实现 ZBot 的接触守卫原地交替踏步。模型资源复制自 `zbot_rl_unilab` 的 MuJoCo 场景；运行时不依赖该强化学习项目。`assets/provenance.json` 保留了模型来源和关节信息。

## 当前稳定踏步入口

当前版本优先保证接触和换脚稳定，不要求固定步频。`--frequency` 指定初始动作节奏；如果搜索需要更慢的动作，后续保留这个周期，动作周期最长 4 秒。每步另执行 0.5 秒双脚稳定阶段，确认速度降低、下一只支撑脚已承重，再换脚。

```bash
# 重新优化轨迹，并保存 MP4 和可重复回放的 NPZ
./replay_stepping.sh --recompute
# 已有缓存时直接回放，无需重新计算 MPC
./replay_stepping.sh --viewer
# 查看汇总指标（不渲染视频）
./run.sh --frequency 0.5 --seconds 20 --candidates 256 --iterations 24 --threads 8
```

回放文件是 `dist/zbot_stable_stepping.mp4` 和 `dist/zbot_stable_stepping.npz`。也可用 `./visualize.sh --replay dist/zbot_stable_stepping.npz --viewer` 直接加载测量轨迹。空格暂停/继续，`N` 单帧前进，`R` 从头播放。橙色表示计划摆动脚，蓝色表示计划支撑脚，接触点显示实际接触。MP4 从 200 Hz 测量轨迹中抽帧，以约 50 fps 导出。

仓库附带当前最佳回放及 `dist/zbot_stable_stepping.json` 指标：固定种子 4、256 条候选、24 次迭代跑满 20 秒，完成 5 次交替换脚；每步离地峰值至少 10.2 mm，连续高于 5 mm 的时间至少 0.19 秒。每个 5 ms 物理步检查下，双脚腾空、支撑丢失、弱单脚支撑和两脚碰撞均为 0 帧；最小单脚支撑力 15.32 N，最大机身水平漂移 66.4 mm。本机离线规划耗时约 352 秒。该结果尚不保证任意种子或长期稳定：种子 7 在完成第一步后因下一步搜索无可行候选而停止。当前保留此版本，不再继续调参。

本版主要调整：

- 控制与接触检查均为 **5 ms**，覆盖每个 MuJoCo 物理子步；不会只用 20 ms 的端点判断支撑。
- 重心转移指令持续到落脚阶段，抬脚代价按实际保持阶段计算。有效踏步要求摆动脚在转移阶段结束后连续至少 0.1 秒高于地面 5 mm。
- 优化落地关节补偿，并在下一步继续保持，避免每次换脚都撤掉已有的平衡姿态。搜索可以从上一步、基础动作和固定支撑脚的运动学姿态分别重新开始；运动学姿态仍须通过完整动力学检查。
- 预测并执行落地后的双脚稳定段。交接要求双脚承重、至少连续 60 ms 接触；自由根线速度不超过 0.03 m/s、角速度不超过 0.15 rad/s、关节速度范数不超过 0.2 rad/s。
- 转移阶段允许由上一只脚继续承重；进入抬脚阶段后要求计划支撑脚持续接触。所有阶段禁止双脚同时离地，单脚支撑法向力至少 15 N，并检查两脚防碰撞。实际执行检查失败或搜索找不到可行轨迹时，会停止并报告原因。
- 默认关闭尚未适配的前进步行 PPO 先验；`--rl-prior` 可显式开启实验候选。开启先验不会额外消耗普通候选的随机采样序列。

这是离线规划和物理仿真：5 ms 是仿真控制与检查粒度，**不是实时 MPC 的计算耗时**。一次完整优化可能需要数分钟。`support_contact_loss_frames` 统计抬脚后的支撑丢失或任何阶段的双脚腾空；`transfer_contact_switch_frames` 单独记录转移阶段仍由上一只脚承重的情况。

## 历史基线与单脚抬腿实验记录

以下保留早期版本的设计说明和测量记录。此前踏步主要检查 20 ms 控制节点；下面的旧数值不作为当前 5 ms 版本的验收结果。

## 双脚站立到单脚抬腿

运行 `./run_single.sh --threads 8 --trajectory dist/double_to_single.npz`。入口从 `stand` keyframe 开始，先用已有步级 MPC 规划 1.3 秒的重心转移，再根据实际 MuJoCo 状态用 `mjbatch` 采样 0.3 秒的抬脚控制。每条候选都检查支撑接触、单脚支撑力、足间距离和机身高度，执行时再次检查。NPZ 包含每 20 毫秒测得的 `qpos`、`qvel`、`ctrl`、双脚接触与离地高度。

用 `./visualize.sh --single-support --viewer --threads 8` 交互回放，或用 `./visualize.sh --single-support --threads 8` 导出 `dist/zbot_single_support.mp4`。画面中橙色为抬起的右脚，蓝色为左支撑脚。

目标直线构型保存在 `straight_up` keyframe：六个关节均为零，右脚位于左脚正上方，高度约 0.583 米，朝上轴竖直分量为 1。过渡过程中右脚姿态完全自由，不要求脚底平行地面。最终判定要求右脚朝上轴竖直分量至少 0.98、右脚本体高度至少 0.55 米、关节角范数不超过 0.15 弧度。直接从站立位姿插值到此构型会使左脚失去支撑并跌倒，因此不能当作有效控制轨迹。

抬脚 MPC 对预测中每个 5 毫秒 MuJoCo 子步的单脚支撑状态检查 CoP/ZMP，要求它位于左脚足底内。实际执行时，双脚承重按双脚足底凸包检查，单脚承重按实际承重脚足底检查；越界立即停止。重心转移阶段可能短暂由右脚单独承重，抬脚阶段及终态要求左脚支撑。预测筛选还要求抬脚时左脚法向力至少 15 N。

固定种子 4、256 条抬脚候选、40 次迭代的本机结果：1.6 秒末进入左脚单脚支撑，右脚在最后 100 毫秒始终离地，最低间隙约 4.2 厘米，抬脚阶段左脚最小法向力超过 15 N，5 毫秒采样下最小 ZMP 支撑裕度约 5 微米。右脚本体高度约 0.114 米，朝上轴竖直分量约 0.11；`straight_goal_reached` 为 `false`。**朝天直线构型和长时间静态单脚平衡尚未实现**。`./run_single.sh --require-straight --threads 8` 会针对这个最终目标返回失败，而不会把短时抬脚当成完成。

## 已实现

- 使用 `mjbatch` 对 13 维位置、12 维速度、6 个位置执行器的 ZBot 做并行 MuJoCo rollout；可读取身体与双脚位置、接触传感器，并限制关节位置目标的变化速度。
- 使用 `implicitfast` 积分器稳定运行原始站立位姿，并启用两次 MuJoCo NoSlip 后处理迭代以抑制软接触模型的缓慢滑动。
- 接触守卫的变频状态机：目标落地间隔为 `T = 1/f`。默认 0.5 Hz 的慢速模式每步先用约 0.9 秒转移重心，再用约 0.3 秒抬脚、0.5 秒保持、0.3 秒落脚。只有摆动脚连续至少 0.1 秒高于地面 3 mm、且最终双脚接触，才算有效换脚。若较快目标节拍没有安全候选，规划器可把当前一步延长，最长 2 秒。
- 参考经典 [Kajita 等人的 ZMP 预览控制](https://people.csail.mit.edu/katiebyl/kb/DW2008/papers_of_tangential_interest/kajita03.pdf)，先根据未来两步的支撑脚位置规划双脚支撑与单脚支撑的 ZMP 目标，再用线性倒立摆的发散分量倒推有界的质心参考。每次落地后，`mjbatch` 采样 MPC 从测量状态出发，优化独立的重心转移和抬脚关节脉冲，并跟踪质心与 CoP/ZMP 参考。预测覆盖一步和 0.5 秒站立恢复。
- MPC 在预测的 20 ms 控制节点上筛除支撑脚失去接触、摆动脚离地时支撑力低于 15 N、摆动脚没有持续抬起、或双脚间隙不足 2 mm 的候选。落地还必须双脚接触；常规步的自由根广义速度与关节速度范数上限均为 1.5，2 秒慢速步按周期平方根放宽，同时仍在代价中压低落地速度，避免把预测中额外 0.5 秒才能稳定的高速状态直接交给下一步。执行时再次检查支撑力。找不到候选时安全停止并输出 `terminated_early` 和 `stop_reason`，不会继续执行已知不安全的轨迹。
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

上面的 1 Hz 命令用于较快节拍的实验。固定种子 4 的本机参考运行可跑满 10 秒并完成 8 次有效换脚；为保持接触安全，部分步会由目标 1 秒自动延长，最长规划周期为 1.5 秒。

## 规划效率与质心雅可比

较快的实验配置可用 `./run.sh --frequency 0.5 --seconds 4.2 --candidates 128 --iterations 8 --threads 8`。本机同一种子下，它用约 4.8 秒完成 4.2 秒仿真及两次换脚，单脚支撑最小法向力 15.97 N；默认的 256 条候选、16 次迭代约需 18 秒，最小法向力 20.8 N。较少的候选降低了安全裕量，64 条候选、8 次迭代在初始状态找不到安全踏步。

运行 `uv run python -m zbot_mpc_mjbatch.jacobian_probe` 可复测 MuJoCo `mj_jacSubtreeCom` 的局部预测能力。在初始站立位姿附近，对 128 条关节脉冲，0.20 秒时质心相对支撑脚位移的 x/y 预测相关系数为 0.552/0.128；到 1.40 秒为 -0.193/-0.151。此时接触模式和自由根姿态已变化，站立位姿的运动学雅可比不能代替控制量到未来接触状态的导数。直接用它预筛 MPC 候选在实测中导致原本可完成的两步提前失败，因此当前控制器没有启用这种预筛。若继续做导数型 MPC，需在每个接触模式下沿名义轨迹计算 MuJoCo 的[状态转移导数](https://mujoco.readthedocs.io/en/latest/computation/#derivatives)，再用完整 rollout 复核接触切换。

`visualize.sh` 先用同一套 MPC 和状态机计算闭环轨迹，再通过 MuJoCo 渲染和 `ffmpeg` 导出 `dist/zbot_stepping.mp4`。脚本在 Linux 默认使用 EGL，在 macOS 默认使用 CGL；也可通过 `MUJOCO_GL` 指定后端。橙色为摆动脚，蓝色为支撑脚，地面显示接触点。可用 `--output result.mp4` 指定文件、`--speed 0.5` 慢放、`--width 1280 --height 720` 调整分辨率；宽高须为偶数。系统需安装 `ffmpeg`。

`./replay_stepping.sh` 使用固定种子重算 1 Hz、10 秒的 MPC 踏步轨迹并写入 `dist/zbot_mpc_1hz_stepping.mp4`；追加 `--viewer` 可直接打开 MuJoCo 窗口回放，追加 `--speed 0.5` 可导出半速视频。该脚本默认关闭尚未适配的前进步行 PPO 先验，以复现当前能跑满 10 秒的 MPC 基线。它是仿真回放入口；当前回放虽完成 8 次换脚，但实测步间隔最长 1.5 秒、离地峰值仅约 4.6 mm，因此不能视为已达到严格 1 Hz 且摆动脚充分离地的目标。

加 `--viewer` 可在 MuJoCo 交互窗口中循环回放同一条轨迹，保留实际关节位置、速度和控制量。空格暂停或继续、`N` 单帧前进、`R` 从头播放，关闭窗口退出；`--once` 播放一遍后自动关闭。Viewer 需要图形桌面；在 macOS 上脚本会自动使用 `mjpython` 启动。MP4 模式仍可用于无桌面环境。

`--frequency` 的有效范围是 0.5–2 Hz，表示目标节拍；`maximum_planned_step_duration_s` 显示安全筛选要求的最长实际规划周期。`--candidates` 和 `--iterations` 控制优化工作量。若提前停止，`run.sh` 会输出已完成段的指标并以非零状态退出。站立 smoke 可传入 `--num-sims 64 --seconds 2 --threads 8`。

`run.sh` 的 `minimum_single_support_com_margin_m` 与 `minimum_single_support_cop_margin_m` 仅统计摆动脚离地且支撑脚接触的时刻；如果没有这样的时刻，显示 `null`。CoP 的法向力不足 1 N 时不定义，也不计入裕度。

本机参考运行（0.5 Hz、4.2 秒、256 条候选）完成 2 次有效换脚，支撑脚失去接触、弱单脚支撑和两脚碰撞均为 0 帧，最低机身高度约 0.253 m。1 Hz、10 秒、固定种子 4 的运行完成 8 次换脚，最小单脚支撑力为 15.45 N，最大水平漂移为 0.0835 m；落地最大线速度、角速度和关节速度范数分别为 0.1512 m/s、1.0251 rad/s 和 1.3363 rad/s。延长到更多步或更换种子时若找不到安全候选，仍会提前停止。

抬脚与支撑的逐步指标包括：完成步骤中最低的抬脚峰值间隙 `minimum_step_peak_clearance_m`、连续高于 3 mm 的最短时长 `minimum_step_3mm_clearance_run_s`、保持阶段的最低支撑接触比例 `minimum_step_support_contact_fraction`、全阶段的 `minimum_step_support_contact_fraction_all_phases`、以及最大支撑脚水平位移 `maximum_step_support_xy_motion_m`。`maximum_landing_linear_speed_m_s`、`maximum_landing_angular_speed_rad_s` 和 `maximum_landing_joint_speed_rad_s` 报告所有已完成步骤的落地速度。`minimum_single_support_force_n` 是摆动脚离地时支撑脚承受的最小法向力；`weak_single_support_frames` 统计低于 15 N 的控制帧。`support_contact_loss_frames` 和 `maximum_support_xy_motion_all_frames_m` 包含未完成的最后一步。`minimum_foot_foot_distance_m` 与 `foot_foot_contact_frames` 检查两脚防撞。

依赖 `mjbatch` 通过 `../../AI/mjbatch` 的本地源码安装；若项目被移至其他机器，需相应调整 `pyproject.toml` 的路径来源，或改从包索引安装。

## 后续验证

1. 对 0.5–2 Hz 做受扰动和不同初始状态的验证，尤其检查 2 Hz 的落地延迟与短暂离地、低频长时间保持，以及 10 秒内约 0.1 m 级的水平漂移。
2. 加入步内短视野反馈和机身姿态约束，进一步降低受扰动时的漂移。
3. 在 Motrix 中复测，识别两种物理后端的接触差异。

本机初测：128 条候选、100 个控制节点（2 秒）、每节点 4 个 MuJoCo 子步，仅物理推进约需 101 毫秒，超过 20 毫秒控制周期；因此当前 MPC 在步边界做规划，并不满足 20 毫秒实时控制周期。
