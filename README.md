# 无人机路径规划仿真系统

## 项目概述

这是一个综合的无人机（UAV）路径规划仿真和强化学习训练系统，集成了传统规划算法（A*）和深度强化学习（PPO）方法。系统支持复杂环境建模、风场模拟、电池管理、任务执行和结果分析。

## 核心功能

### 1. 传统仿真模块
- **路径规划**：基于A*算法的最优路径规划
- **物理引擎**：无人机动力学模拟（加速度、速度、位置更新）
- **电池管理**：电耗模型、电池管理策略
- **状态估计**：位置、速度、电池状态等实时估计
- **风场模型**：支持均匀风、时变风、空间变化风等多种风场模型
- **地图管理**：障碍物检测、碰撞避免

### 2. 强化学习模块
- **RL环境**：基于OpenAI Gym框架的自定义无人机环境
- **PPO训练**：使用Stable-Baselines3实现的PPO算法
- **多阶段训练**：支持课程学习和渐进式难度提升
- **模型管理**：检查点保存、最佳模型选择、评估

### 3. 分析与可视化
- **基准测试**：固定测试套件、长距离任务评估
- **性能指标**：任务完成率、能耗效率、路径长度等
- **动画渲染**：任务执行过程的可视化和动画生成
- **结果分析**：CSV格式的测试结果导出和对比分析

### 4. 用户界面
- Web UI支持（通过 ui/drone_ui.py）
- 实时参数配置
- 结果可视化

## 项目结构

```
project1/
├── main.py                          # 传统仿真主程序
├── launcher.py                      # 统一启动器
├── check_coords.py                  # 坐标检查工具
├── smoke_test.py                    # 快速测试脚本
│
├── configs/                         # 配置模块
│   ├── config.py                   # 仿真参数配置
│   └── eval_config.py              # 评估参数配置
│
├── core/                           # 核心算法模块
│   ├── physics.py                  # 物理引擎
│   ├── estimator.py                # 状态估计器
│   ├── planner.py                  # A*路径规划器
│   └── battery_manager.py          # 电池管理
│
├── environment/                    # 环境模块
│   ├── map_manager.py             # 地图与障碍物管理
│   └── wind_models.py             # 风场模型
│
├── simulation/                     # 仿真执行
│   └── mission_executor.py        # 任务执行器
│
├── rl_env/                        # RL环境
│   └── drone_env.py               # 自定义RL环境
│
├── rl_training/                   # 强化学习训练
│   └── train_ppo.py               # PPO训练脚本
│
├── adapters/                      # 适配器
│   └── rl_adapter.py             # RL与仿真的适配层
│
├── analysis/                      # 分析与基准测试
│   ├── benchmark_fixed_suite.py   # 固定测试套件
│   ├── benchmark_long_distance.py # 长距离基准测试
│   ├── mission_metrics.py         # 性能指标计算
│   └── render_case_studies.py     # 案例渲染
│
├── models/                        # 模型存储
│   ├── mission_models.py          # 数据模型类
│   ├── ppo_drone_stage1_obs31_run2_*  # 訓練的PPO模型
│   └── ...
│
├── ui/                           # 用户界面
│   └── drone_ui.py              # Web UI
│
├── utils/                        # 工具库
│   ├── visualizer_core.py       # 可视化核心
│   ├── visualizer.py            # 可视化工具
│   └── animation_builder.py     # 动画生成
│
├── logs/                         # 训练日志
│   ├── stage1_obs31_run2/
│   ├── stage2_obs31_run2/
│   └── stage3_obs31_run1/
│
├── results/                      # 测试结果
│   ├── benchmark_stage1_model_on_stage1/
│   ├── benchmark_stage2_model_on_stage2/
│   └── ...
│
├── mission_outputs/              # 任务输出
│
├── tests/                        # 单元测试
│   ├── test_physics.py
│   ├── test_map_and_planner.py
│   ├── test_mission_executor.py
│   └── ...
│
└── requirements.txt              # 依赖包列表
```

## 安装指南

### 前置要求
- Python >= 3.8
- 推荐使用 Conda 或 venv 创建虚拟环境

### 安装步骤

1. **克隆或下载项目**
   ```bash
   cd project1
   ```

2. **使用pip安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装项目（可选）**
   ```bash
   pip install -e .
   ```

### 快速测试
```bash
python smoke_test.py  # 运行快速测试
```

## 使用指南

### 1. 运行传统仿真
```bash
python main.py
```
- 执行单次任务仿真
- 需要在 `configs/config.py` 中配置参数

### 2. 强化学习训练
```bash
python launcher.py rl
# 或直接运行
python rl_training/train_ppo.py
```
- 启动PPO模型训练
- 支持多阶段课程学习
- 自动保存检查点和评估结果

### 3. 基准测试
```bash
python analysis/benchmark_fixed_suite.py   # 固定测试套件
python analysis/benchmark_long_distance.py # 长距离测试
```
- 测试多个模型或策略
- 生成性能对比报告

### 4. 启动UI
```bash
python launcher.py ui
# 或直接运行
python ui/drone_ui.py
```

### 5. 运行测试套件
```bash
python launcher.py test
# 或使用 pytest
pytest tests/
```

### 6. 使用启动器
```bash
python launcher.py --help

使用示例：
  python launcher.py sim          # 运行传统仿真
  python launcher.py rl           # 训练RL模型
  python launcher.py ui           # 启动Web界面
  python launcher.py test         # 运行测试
  python launcher.py install      # 安装依赖
```

## 主要模块说明

### 配置系统 (`configs/`)
- **SimulationConfig**: 仿真全局参数
  - 环境参数：地图大小、障碍物设置
  - 无人机参数：最大速度、加速度、质量
  - 电池参数：容量、耗电率
  - 规划参数：A*网格分辨率、安全距离

### 核心算法 (`core/`)
- **PhysicsEngine**: 运动学与动力学模型
- **StateEstimator**: 实时状态估计（位置、速度等）
- **AStarPlanner**: 基于栅格的A*最优路径规划
- **BatteryManager**: 电池管理与能耗计算

### 环境模型 (`environment/`)
- **MapManager**: 栅格地图管理、碰撞检测
- **WindModelFactory**: 多种风场模型
  - UniformWind: 均匀风场
  - TimeVaryingWind: 随时间变化的风
  - SpatialWind: 空间变化的风

### RL训练 (`rl_training/`)
- **train_ppo.py**: PPO算法的完整训练流程
  - 多环境并行（vectorized）
  - 课程学习支持
  - 定期评估和检查点保存
  - 学习率调度

### 分析与基准 (`analysis/`)
- **BenchmarkFixedSuite**: 固定难度的测试套件
- **BenchmarkLongDistance**: 超长距离任务评估
- **MissionMetrics**: 计算性能指标
  - 完成率、任务时间、能耗效率
  - 路径长度、平均速度等

## 关键配置参数

在 `configs/config.py` 中修改：

```python
@dataclass
class SimulationConfig:
    # 环境
    map_width: float = 1000           # 地图宽度
    map_height: float = 1000          # 地图高度
    obstacle_ratio: float = 0.2       # 障碍物占比
    
    # 无人机
    max_velocity: float = 30          # 最大速度 (m/s)
    max_accel: float = 2              # 最大加速度 (m/s²)
    mass: float = 1.5                 # 质量 (kg)
    
    # 电池
    battery_capacity: float = 100     # 电池容量 (Wh)
    power_consumption_rate: float = 50  # 耗电率 (W)
    
    # 规划
    grid_resolution: float = 10       # A*网格分辨率
    safety_margin: float = 5          # 安全距离
    
    # 风场
    wind_model_type: str = "uniform"  # 风场类型
    wind_speed: float = 5             # 风速 (m/s)
```

## 模型和结果

### 已训练的模型
- `ppo_drone_stage1_obs31_run2_best`: 第一阶段最优模型
- `ppo_drone_stage2_obs31_run2_best`: 第二阶段最优模型
- `ppo_drone_stage3_obs31_run1_best`: 第三阶段最优模型

### 测试结果
- `results/benchmark_stage1_obs31/`: 第一阶段基准测试
- `results/benchmark_stage2_obs31/`: 第二阶段基准测试
- `results/benchmark_stage3_obs31/`: 第三阶段基准测试

每个结果目录包含 `summary.csv`，包括：
- 任务完成率
- 平均任务时间
- 平均能耗
- 路径效率指标

## 开发和测试

### 运行单元测试
```bash
pytest tests/ -v
```

### 代码风格
- 遵循 PEP 8 规范
- 使用类型提示（Python 3.8+）

### 添加新的风场模型
参考 `environment/wind_models.py`，继承 `BaseWindModel` 类：

```python
class CustomWindModel(BaseWindModel):
    def get_wind_at(self, x: float, y: float, t: float) -> Tuple[float, float]:
        # 返回 (wind_x, wind_y)
        pass
```

### 添加新的基准测试
参考 `analysis/benchmark_fixed_suite.py` 的结构：

```python
def run_benchmark(config, num_episodes=10):
    # 实现测试逻辑
    results = []
    for episode in range(num_episodes):
        # ...
    return results
```

## 系统要求

- **CPU**: 多核处理器推荐（用于并行RL训练）
- **内存**: 最少8GB，推荐16GB+
- **GPU**: 可选，但训练速度会显著提升（推荐 NVIDIA CUDA）
- **存储**: 预留5GB+用于模型和日志

## 常见问题

### Q: 如何训练自己的模型？
A: 修改 `configs/config.py` 中的参数，然后运行：
```bash
python rl_training/train_ppo.py
```

### Q: 如何加载已有的模型进行评估？
A: 查看 `analysis/benchmark_fixed_suite.py` 的 `load_model_for_benchmark()` 函数。

### Q: 如何自定义任务场景？
A: 在 `configs/config.py` 中修改地图参数、风场类型、起点和终点等。

### Q: 程序崩溃了怎么办？
A: 检查 `logs/` 目录中的错误日志，查看是否有内存不足或路径问题。

## 贡献

欢迎提交问题和改进建议。

## 许可证

项目许可信息（如有）

## 联系方式

如有问题或建议，请联系项目维护者。

---

**最后更新**: 2026-04-02

**版本**: v1.0
