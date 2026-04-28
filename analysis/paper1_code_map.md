# 论文一实验代码地图

这份文档从“实验设计可直接使用”的角度，梳理当前仓库里与风场建模、风险建模、能耗计算和实验脚本最相关的代码。

## 1. 目前代码主线

当前系统已经形成一条比较完整的链路：

`地图/地形 -> 风场生成 -> 状态估计 -> 风险/TKE估计 -> 4D A* 或 RL 执行 -> 功率/能耗/电池约束 -> benchmark/ablation/case study 输出`

如果只看论文一最核心的单机问题，最关键的调用链是：

`SimulationConfig -> MapManager -> WindModelFactory/SlopeWindModel -> StateEstimator -> PhysicsEngine -> AStarPlanner -> MissionExecutor -> analysis/*.py`

如果要扩展到群体协同/FANET，则在单机链路上再往外包一层：

`SwarmMissionExecutor`

## 2. 风场建模相关代码

### 2.1 风场核心实现

- [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):14
  定义 `BaseWindModel` 抽象接口，所有风场模型都要实现 `get_wind(...)`。

- [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):31
  `StatelessStormCell` 定义了时空风暴单元，支持按绝对时间查询风暴中心、是否激活、某点风速贡献。

- [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):73
  `StormWindManager` 负责预生成整个任务时域内的风暴序列，核心受 `wind_seed` 控制，可复现实验。

- [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):161
  `SlopeWindModel` 是当前真正使用的主风场模型。

### 2.2 当前风场模型的组成

`SlopeWindModel.get_wind(...)` 当前由 5 部分叠加：

1. 背景风
   [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):182

2. 坡度风
   [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):185

3. 高度对数风廓线修正
   [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):197

4. 移动风暴叠加
   [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):202

5. 最大风速裁剪
   [environment/wind_models.py](/abs/path/c:/Users/20340/Desktop/大创/project1/environment/wind_models.py):208

### 2.3 风场里已经可控的实验变量

这些参数都集中在 [configs/config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py)：

- 背景风: `env_wind_u`, `env_wind_v`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):113

- 坡度风强度: `k_slope`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):115

- 风速上限: `max_wind_speed`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):116

- 是否启用风暴: `enable_storms`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):121

- 风暴数量/尺度/寿命/强度
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):122
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):123
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):126
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):127

- 风场随机种子: `wind_seed`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):128

- 时变背景风周期/方向扰动/速度扰动
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):211
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):212
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):213

- 执行期阵风
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):218
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):219
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):220
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):221
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):222
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):224

### 2.4 风场估计与风险映射

- [core/estimator.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/estimator.py):37
  `get_wind(...)` 会把地图高程、粗糙度、坡度送进风场模型，并可叠加单机阵风和观测噪声。

- [core/estimator.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/estimator.py):80
  `get_tke(...)` 当前使用参数化方法，把风切变、粗糙度尾流项、坡度扰动项组合成 TKE。

- [core/estimator.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/estimator.py):109
  `get_risk(...)` 把 TKE 映射成 `p_crash`，这是规划器里的风险代价来源。

对应核心风险参数：

- `fatal_crash_penalty_j`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):136

- `tke_shear_coeff`, `tke_wake_coeff`, `tke_slope_coeff`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):138
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):139
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):140

## 3. 能耗计算相关代码

### 3.1 功率模型

- [core/physics.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/physics.py):94
  `power_for_speed(v_air_mag)` 使用空气相对速度的三次方阻力功率模型，再加基础功率 `base_power`。

- [core/physics.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/physics.py):110
  `estimate_power_from_vectors(...)` 用地速向量和风速向量求相对空速，再叠加爬升功率。

### 3.2 单段与整条路径能耗

- [core/physics.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/physics.py):198
  `estimate_segment_energy(...)` 输出单段的能耗、时间、平均功率。

- [core/physics.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/physics.py):230
  `estimate_path_energy(...)` 沿路径中点采样风场，累计整条路径能耗。

### 3.3 最大功率与可行速度

- [core/physics.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/physics.py):140
  `find_feasible_speed(...)` 在逆风或高负载下，尝试降低速度直到功率不超过 `max_power`。

这对论文实验很重要，因为它决定了“强风下是绕行还是硬顶”的边界。

### 3.4 电池与能量约束

- [core/battery_manager.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/battery_manager.py):13
  `get_min_reserve_energy_j()` 给出保底电量。

- [core/battery_manager.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/battery_manager.py):25
  `can_consume(...)` 决定某段能量能否被安全消耗。

- [core/battery_manager.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/battery_manager.py):38
  `is_path_feasible(...)` 用于重规划前先做整条路径电量可行性判断。

对应关键参数：

- `base_power`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):160

- `drag_coeff`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):162

- `max_power`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):164

- `cruise_speed_mps`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):196

- `battery_capacity_j`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):198

- `reserve_energy_ratio`
  [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):199

## 4. 规划与执行是怎么把风场和能耗用起来的

### 4.1 4D 风险感知 A*

- [core/planner.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/planner.py):91
  `calculate_cost(...)` 是目前论文一最关键的代价函数。

当前代价由两部分组成：

1. `energy_joules`
2. `risk_cost = fatal_crash_penalty_j * p_crash * k_wind`

规划器还显式考虑了：

- 到达未来时刻的风场查询
  [planner.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/planner.py):115
  [planner.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/planner.py):138

- 未来时刻的风险查询
  [planner.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/planner.py):152

- 超功率即不可行
  [planner.py](/abs/path/c:/Users/20340/Desktop/大创/project1/core/planner.py):141

`planner_time_mode` 已经支持两种时域设定：

- `4d`
- `frozen_3d`

定义在 [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):235，校验在 [config.py](/abs/path/c:/Users/20340/Desktop/大创/project1/configs/config.py):359。

### 4.2 单机任务执行器

- [simulation/mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/mission_executor.py):40
  `execute_mission(...)` 会先规划无风路径、再规划考虑风的路径，然后周期性重规划。

- [simulation/mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/mission_executor.py):117
  重规划后先估算整条路径能耗，再做电池可行性判断。

- [simulation/mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/mission_executor.py):198
  `_advance_along_path(...)` 执行真实飞行段，逐段统计功率、风险和累计能耗。

这里的失败原因已经比较适合论文统计：

- `overload`
- `storm_risk_too_high`
- `battery_depleted`
- `planner failed to find a path`

## 5. 群体/FANET 相关代码

如果论文一先聚焦单机，可以把这部分作为扩展储备；如果你们准备把“协同预警”放进第一篇，那这部分已经有可跑基础。

- [simulation/swarm_mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/swarm_mission_executor.py):87
  `execute_mission(...)` 是群体主入口。

- [simulation/swarm_mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/swarm_mission_executor.py):326
  `_advance_master_step_astar(...)` 表示主机按 A* 参考执行。

- [simulation/swarm_mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/swarm_mission_executor.py):392
  `_advance_master_step_rl(...)` 表示主机按 RL 执行。

- [simulation/swarm_mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/swarm_mission_executor.py):827
  `_estimate_master_step_telemetry(...)` 直接给出主机功率和风险时间序列。

- [simulation/swarm_mission_executor.py](/abs/path/c:/Users/20340/Desktop/大创/project1/simulation/swarm_mission_executor.py):893
  `_get_execution_wind(...)` 会把真实风场和执行期阵风合并。

这个模块适合做的实验更偏：

- 拓扑对首次预警距离的影响
- shield 机制对峰值功率和峰值风险的影响
- 支援机/中继机对重规划时效的影响

## 6. 已有实验脚本和它们分别适合做什么

### 6.1 单机基准对比

- [analysis/benchmark_fixed_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/benchmark_fixed_suite.py):268
  `run_benchmark(...)`

适合回答：

- A*、Teacher、RL residual 在固定任务集上的成功率差异
- 成功样本的平均能耗、单位距离能耗差异
- 失败类型构成

输出已经比较完整：

- `raw_runs.csv`
- `summary.csv`
- `control_success_energy_dual_axis.png`
- `control_failure_breakdown_stacked_bar.png`

### 6.2 长距离能耗趋势

- [analysis/benchmark_long_distance.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/benchmark_long_distance.py):62
  `generate_and_save_tasks()`

- [analysis/benchmark_long_distance.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/benchmark_long_distance.py):196
  `run_long_distance_benchmark()`

适合回答：

- 距离增加时不同方法成功率如何变化
- 单位距离能耗是否随距离带变化
- 长航程下谁先触发电池瓶颈/规划失败/风险失败

### 6.3 消融总控脚本

- [analysis/run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):108
  `run_exp1_control`

- [run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):124
  `run_exp2_observation`

- [run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):244
  `run_exp3_topology`

- [run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):315
  `run_exp4_shield`

- [run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):421
  `run_exp5_planner_time`

其中对论文一最有直接价值的是：

1. `exp1_control`
   基本对照组
2. `exp5_planner_time`
   直接比较 4D 规划和冻结时刻 3D 规划
3. `exp2_observation`
   如果论文一要加入 RL 观测设计，这是现成消融入口

### 6.4 案例图渲染

- [analysis/render_case_studies.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/render_case_studies.py):54
  `render_custom_case(...)`

适合论文中做：

- 单案例可视化
- 2D 轨迹图
- 高度剖面图
- 功率/能耗图
- GIF 动态演示

### 6.5 CSV 复画图

- [analysis/plot_benchmark_from_csv.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/plot_benchmark_from_csv.py):74
  `load_metrics(...)`

- [analysis/plot_benchmark_from_csv.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/plot_benchmark_from_csv.py):83
  `plot_2x2(...)`

这份脚本适合在你后面反复调图时使用，不用重新跑实验，只要有 `raw_runs.csv` 或 `summary.csv` 就能出 2x2 对比图。

## 7. 目前已经具备的可用指标

从代码和结果结构看，当前最稳定、最适合论文表格与图的指标有：

- 任务成功率 `success`
- 总能耗 `total_energy_used_j`
- 单位距离能耗 `energy_kj_per_km`
- 任务时间 `total_mission_time_s`
- 路径长度 `path_length_m`
- 峰值功率/平均功率
- 风险时间序列 `risk_history`
- 重规划次数 `total_replans`
- 失败原因分桶

群体实验额外还有：

- `first_warning_distance_m`
- `warning_count`
- `master_power_history_w`
- `master_risk_history`

## 8. 论文一最建议优先做的实验设计

如果论文一主打“时空风场感知下的安全节能路径规划”，我建议优先收敛到下面三组。

### 8.1 主实验

比较对象：

- 传统几何 A*
- 风险/风场未显式建模的参考方法
- 你们的 4D 风场风险感知方法

核心指标：

- 成功率
- 平均总能耗
- 单位距离能耗
- 平均任务时间
- 失败原因构成

直接可复用脚本：

- [analysis/benchmark_fixed_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/benchmark_fixed_suite.py):268

### 8.2 关键消融

推荐至少做这三项：

1. `4d` vs `frozen_3d`
   对应时变风场是否真的有价值
   [analysis/run_ablation_suite.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/run_ablation_suite.py):421

2. 风险代价开/关
   可通过 `k_wind` 或 `fatal_crash_penalty_j` 做

3. 风场复杂度分层
   逐步打开：
   背景风 -> 坡度风 -> 时变风 -> 风暴 -> 阵风

### 8.3 长距离扩展实验

适合作为“扩展验证”：

- 6km/8km/10km/12km 任务带
- 观察成功率下降曲线
- 观察 `kJ/km` 是否更优

直接入口：

- [analysis/benchmark_long_distance.py](/abs/path/c:/Users/20340/Desktop/大创/project1/analysis/benchmark_long_distance.py):196

## 9. 我对当前代码状态的判断

### 9.1 优点

- 风场、风险、能耗、电池、规划、执行、分析脚本已经打通。
- 风场参数和实验种子基本可控，适合做可复现实验。
- 已经有单机和群体两条实验线，论文空间比较大。
- benchmark 和 ablation 输出格式已经足够接近论文图表需求。

### 9.2 当前最需要注意的点

1. 风场模型目前仍是参数化工程模型，不是高保真 CFD。
   这没有问题，但论文表述里最好强调“physics-informed / parametric wind field”而不是过度宣称真实气象重建。

2. 风险模型是由 TKE 到 `p_crash` 的映射。
   论文里需要把它定义成“任务层风险代理指标”或“极值响应启发式风险模型”。

3. 能耗模型当前以阻力功率加爬升功率为主。
   对固定翼/多旋翼的具体机型映射还不够细，但做相对比较实验是够用的。

4. 仓库里有一些中文注释编码已经出现乱码。
   不影响运行，但后续如果要交叉协作、补论文附录或整理开源版本，建议统一一次文件编码。

## 10. 你接下来最省力的推进方式

建议你先把论文一收敛成下面这个结构：

1. 问题定义
   时变风场下的安全节能路径规划

2. 方法主体
   时变风场 + TKE 风险代理 + 风险能耗联合代价 + 4D A*

3. 实验一
   固定任务基准对比

4. 实验二
   `4d` vs `frozen_3d`

5. 实验三
   长距离扩展验证

6. 案例图
   `render_case_studies.py`

如果后面决定把 RL 也放进论文一，再补：

7. 观测/控制消融
   `exp2_observation`

8. RL vs A* hybrid 对比

## 11. 建议作为后续整理目标的文件

如果我们下一步继续整理，我建议优先处理这几类：

- 给 `configs/config.py` 补一份“论文变量对照表”
- 给 `analysis/` 下脚本统一命名和参数入口
- 把单机实验和群体实验拆成两套更清晰的目录说明
- 清理乱码注释，补一版简洁英文/中文 docstring

