# Project1 Drone Path Planning Simulation

This repository contains a simplified drone path planning simulator with
wind and terrain interaction. 主要模块如下：

- `configs/config.py` – 仿真参数与物理常量。
- `environment/map_manager.py` – 高程数据管理、梯度与粗糙度计算。
- `environment/wind_models.py` – 风场模型基类与实现（当前只有坡度模型）。
- `core/physics.py` – 物理引擎：功率、速度与能耗计算。
- `core/estimator.py` – 状态估计／代理层：提供高度、风速、风险等接口。
- `core/planner.py` – 3D A* 搜索器，代价包含能量消耗与风险。
- `utils/visualizer_core.py` – 绘图工具，可显示路径、风速和粗糙度。
- `main.py` – 演示脚本，执行两种路径并显示结果。
- `utils/visualizer.py` – 旧版本的演示脚本，已被标记为遗留。

## 运行仿真

```powershell
cd c:\Users\20340\Desktop\project1
python main.py
```

程序将先计算“最短距离”路径，再计算“风感知能量优化”路径，并
弹出两张图：高度剖面与俯视地图。

## 单元测试

已添加若干基础测试，覆盖物理模型、风场以及地图/规划器。

```powershell
python -m unittest tests.test_physics tests.test_wind tests.test_map_and_planner -v
```

建议在做修改后运行以确保核心功能不被破坏。

## 扩展与建议

- 可以在 `configs` 中调节 `k_wind`、`risk_factor` 等权重。
- 风场模型可通过 `WindModelFactory` 插件式添加。
- 未来可尝试 RRT/D*、卡尔曼滤波等算法。

> ⚠ 本项目目前仅用于原型开发，无真实传感器输入。
