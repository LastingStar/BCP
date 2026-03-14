# 无人机路径规划仿真系统 - 扩展版本

基于原有的规则规划系统，新增强化学习(RL)和用户界面(UI)功能。

## 🚀 新功能概述

### 1. 强化学习模块 (rl_env/, rl_training/)
- **Gym环境封装**: 将路径规划问题形式化为标准RL环境
- **PPO算法训练**: 使用Stable Baselines3实现PPO算法训练
- **连续动作空间**: 支持航向角、俯仰角、速度的连续控制

### 2. 用户界面模块 (ui/)
- **Web界面**: 基于Streamlit的现代化Web界面
- **实时参数配置**: 动态调整仿真参数
- **3D可视化**: 基于Plotly的交互式3D路径显示
- **多模式支持**: 规则规划、RL推理、对比分析

## 📦 安装依赖

```bash
pip install -r requirements_rl_ui.txt
```

## 🏃‍♂️ 使用指南

### 强化学习训练

```bash
# 训练PPO模型
python rl_training/train_ppo.py

# 或者在代码中自定义参数
from rl_training.train_ppo import train_ppo
from configs.config import SimulationConfig

config = SimulationConfig()
model = train_ppo(config, total_timesteps=1_000_000)
```

### 用户界面启动

```bash
# 启动Web界面
streamlit run ui/drone_ui.py
```

然后在浏览器中访问显示的地址。

## 🏗️ 技术架构

### RL环境设计

**状态空间 (12维)**:
- 无人机位置: [x, y, z]
- 目标位置: [x, y, z]
- 风场信息: [wind_u, wind_v]
- 系统状态: [energy_ratio, time_step, dist_to_goal, heading_error]

**动作空间 (3维连续)**:
- 航向角变化: [-180°, 180°]
- 俯仰角变化: [-45°, 45°]
- 速度比例: [0.1, 1.0]

**奖励函数**:
- 距离奖励: 向目标前进获得正奖励
- 能量惩罚: 消耗能量获得负奖励
- 风险惩罚: 进入高风险区域获得负奖励
- 终止奖励: 成功+1000, 失败-500

### UI功能特性

- **参数配置面板**: 实时调整风场、无人机、规划参数
- **3D可视化**: 交互式地形和路径显示
- **仿真控制**: 开始、重置、导出结果
- **多模式支持**: A*规划、PPO推理、性能对比

## 🎯 扩展路线图

### 近期目标 (1-2个月)
- [ ] 完善RL环境的状态空间和奖励函数
- [ ] 实现PPO模型的推理集成
- [ ] 优化UI的响应速度和可视化效果
- [ ] 添加模型评估和对比功能

### 中期目标 (3-6个月)
- [ ] 支持多智能体路径规划
- [ ] 集成真实气象数据
- [ ] 添加路径后处理优化
- [ ] 实现分布式训练支持

### 长期目标 (6个月+)
- [ ] 开发移动端应用
- [ ] 支持实时气象数据接入
- [ ] 实现云端部署和API服务
- [ ] 添加机器学习模型的在线学习能力

## 📊 性能对比

| 特性 | 原系统 | RL扩展 | UI扩展 |
|------|--------|--------|--------|
| 规划方法 | A*搜索 | PPO策略 | 两者皆可 |
| 动作空间 | 离散26邻域 | 连续3维 | 用户选择 |
| 可视化 | 静态图表 | 策略网络 | 交互式3D |
| 用户友好性 | 命令行 | 编程接口 | Web界面 |
| 计算复杂度 | 中等 | 高(训练) | 中等 |

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统！

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd drone-planning-sim

# 安装依赖
pip install -r requirements_rl_ui.txt

# 运行测试
pytest tests/
```

### 代码规范
- 使用Black进行代码格式化
- 使用Flake8进行代码检查
- 添加适当的类型注解
- 为新功能编写单元测试

## 📄 许可证

本项目采用MIT许可证。</content>
<parameter name="filePath">c:\Users\20340\Desktop\project1\README_RL_UI.md