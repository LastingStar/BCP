# 项目结构说明

```
drone-planning-sim/
├── 📁 configs/                 # 配置模块
│   ├── config.py              # 仿真系统主配置
│   └── __pycache__/
├── 📁 core/                   # 核心算法模块
│   ├── estimator.py           # 状态估计器
│   ├── physics.py             # 物理引擎
│   ├── planner.py             # A*路径规划器
│   ├── battery_manager.py     # 电池管理系统
│   └── __pycache__/
├── 📁 environment/            # 环境模拟模块
│   ├── map_manager.py         # 地图管理器
│   ├── wind_models.py         # 风场模型
│   └── __pycache__/
├── 📁 simulation/             # 仿真执行模块
│   ├── mission_executor.py    # 任务执行器
│   └── __pycache__/
├── 📁 analysis/               # 分析工具模块
│   └── mission_metrics.py     # 任务指标分析
├── 📁 models/                 # 数据模型模块
│   └── mission_models.py      # 任务数据结构
├── 📁 utils/                  # 工具模块
│   ├── visualizer_core.py     # 可视化核心
│   ├── animation_builder.py   # 动画构建器
│   └── __pycache__/
├── 📁 tests/                  # 测试套件
│   ├── test_*.py             # 各模块单元测试
│   └── __init__.py
├── 📁 rl_env/                 # 🔄 强化学习环境 (新增)
│   └── drone_env.py          # Gym环境封装
├── 📁 rl_training/            # 🔄 RL训练脚本 (新增)
│   └── train_ppo.py          # PPO算法训练
├── 📁 ui/                     # 💻 用户界面 (新增)
│   └── drone_ui.py           # Streamlit Web界面
├── 📄 main.py                 # 🚀 系统入口
├── 📄 launcher.py             # 🎯 统一启动器
├── 📄 requirements.txt        # 原始依赖
├── 📄 requirements_rl_ui.txt  # 🔄 RL+UI依赖
├── 📄 README.md               # 原项目文档
├── 📄 README_RL_UI.md         # 🔄 扩展功能文档
└── 📄 任务书.md               # 项目任务书
```

## 📂 各模块职责详解

### 🎯 **核心系统**
- **`main.py`**: 系统入口，初始化组件，执行完整仿真流程
- **`configs/config.py`**: 集中配置管理，所有参数在此处调整
- **`launcher.py`**: 统一启动器，支持不同模式的快速启动

### 🧠 **传统规划系统**
- **`core/`**: 核心算法层
  - `planner.py`: 4D A*路径规划，平衡能量和风险
  - `physics.py`: 物理计算引擎
  - `estimator.py`: 状态估计和环境感知
  - `battery_manager.py`: 能量管理

- **`environment/`**: 环境建模
  - `map_manager.py`: DEM地形和碰撞检测
  - `wind_models.py`: 多尺度风场模型

- **`simulation/mission_executor.py`**: 动态任务执行和重规划

### 🔄 **强化学习扩展 (rl_env/, rl_training/)**
- **`rl_env/drone_env.py`**: OpenAI Gym环境封装
  - 状态空间: 位置、目标、风场、能量等12维
  - 动作空间: 航向、俯仰、速度3维连续控制
  - 奖励函数: 距离+能量+风险综合奖励

- **`rl_training/train_ppo.py`**: PPO算法训练框架
  - 支持多进程并行训练
  - 自动模型保存和评估
  - TensorBoard日志记录

### 💻 **用户界面扩展 (ui/)**
- **`ui/drone_ui.py`**: 现代化Web界面
  - 参数配置面板
  - 实时3D可视化
  - 仿真控制和结果分析

### 📊 **分析与工具**
- **`analysis/mission_metrics.py`**: 性能指标计算
- **`utils/visualizer_core.py`**: 静态可视化
- **`utils/animation_builder.py`**: 动态动画生成
- **`models/mission_models.py`**: 数据结构定义

### 🧪 **测试与文档**
- **`tests/`**: 全面的单元测试覆盖
- **`README_RL_UI.md`**: 扩展功能详细文档
- **`requirements_rl_ui.txt`**: 新功能依赖列表

## 🔄 扩展路线图

### Phase 1: 基础RL集成 ✅
- [x] Gym环境封装
- [x] PPO训练框架
- [x] 基础UI界面

### Phase 2: 高级功能 (进行中)
- [ ] RL模型推理集成
- [ ] 多算法对比 (SAC, TD3)
- [ ] UI性能优化

### Phase 3: 生产就绪
- [ ] 分布式训练支持
- [ ] 实时气象数据集成
- [ ] 云端部署和API服务

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements_rl_ui.txt

# 2. 运行传统仿真
python main.py

# 3. 启动Web界面
python launcher.py ui

# 4. 训练RL模型
python launcher.py rl
```</content>
<parameter name="filePath">c:\Users\20340\Desktop\project1\PROJECT_STRUCTURE.md