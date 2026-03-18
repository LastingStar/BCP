> 
>
> ---
>
> # 🚁 4D Spatio-Temporal Risk-Aware UAV Path Planning
> **基于多源物理约束与微气象风险的无人机 4D 时空轨迹规划系统**
>
> ![License](https://img.shields.io/badge/license-MIT-blue.svg)
> ![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
> ![Status](https://img.shields.io/badge/status-Stable%20Release-success)
> ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
>
> 本系统是一个深度耦合了**底层空气动力学、数字高程模型 (DEM)、大气边界层微气象学以及马尔可夫决策过程 (MDP)** 的工业级无人机仿真与导航框架。
>
> 区别于传统的纯几何静态避障，本项目针对**极端恶劣天气（强风切变、移动雷暴）**与**复杂山地地形**，提出了创新的 **4D 时空预测规划算法 (4D Spatio-Temporal A*)**，赋予了无人机拟人化的风险博弈意识与绝境求生能力。
>
> ---
>
> ## ✨ 核心特性 (Key Features)
>
> - 🏔️ **三维数字高程孪生 (3D Terrain Twin)**
>   - 导入真实地理 DEM 数据，精准感知地形梯度，智能区分“耗电爬升”与“省电平飞”。
> - 🌪️ **TKE 极值概率风险映射 (Gust Exceedance Model)**
>   - 首创将**湍流动能 (TKE)**、风速切变与地表粗糙度，通过极值响应理论转化为**连续的坠机概率矩阵 ($P_{crash}$)**。
> - ⏳ **4D 时空前瞻推演 (Spatio-Temporal Planning)**
>   - 突破静态算法“时间冻结”缺陷，在扩展搜索树时同步累加时间 $t$，精确预判并抢占移动雷暴阵列中的“时空安全窗口”。
> - 🛡️ **自适应降级容错机制 (Adaptive Risk Fallback)**
>   - 在禁飞区与雷暴彻底封死去路的死锁绝境下，系统可主动降低安全阈值，生成“高风险突防备用航线”，确保任务底线。
> - 🎬 **全景动态可视化与 Web UI (Interactive Dashboard)**
>   - 内置基于 Streamlit 的可视化看板，支持**一键渲染 4D 动态风暴躲避 GIF** 与 **真 3D 交互航迹图**。
>
> ---
>
> ## 🛠️ 系统架构 (Architecture)
>
> ```text
> 📦 drone-planning-sim
>  ┣ 📂 configs          # 全局配置中心 (物理参数/微气象参数/NFZ坐标)
>  ┣ 📂 core             # 算法核心引擎
>  │ ┣ 📜 planner.py     # 4D 极值风险感知 A* 规划器
>  │ ┣ 📜 physics.py     # 空气动力学能耗模型 (P = P_drag + P_climb)
>  │ ┣ 📜 estimator.py   # TKE 计算与极值概率风险映射
>  │ ┗ 📜 battery_manager.py
>  ┣ 📂 environment      # 环境建模
>  │ ┣ 📜 map_manager.py # DEM 高程解析与高斯梯度滤波
>  │ ┗ 📜 wind_models.py # 时变风场与 Gaussian Decay 移动风暴模型
>  ┣ 📂 simulation       # 动态闭环执行器
>  ┣ 📂 ui               # Streamlit 交互式前端看板
>  ┣ 📂 utils            # 工具链 (GIF 动画合成、B 样条平滑、绘图)
>  ┗ 📜 main.py          # 传统终端启动入口
> ```
>
> ---
>
> ## 🚀 快速开始 (Quick Start)
>
> ### 1. 环境依赖配置
> 建议使用 Conda 创建纯净的虚拟环境：
> ```bash
> conda create -n uav_env python=3.9
> conda activate uav_env
> 
> # 安装核心依赖
> pip install numpy scipy matplotlib opencv-python plotly streamlit
> ```
>
> ### 2. 启动交互式 Web UI（强烈推荐）
> 通过极其友好的 Web 界面，体验所见即所得的 4D 时空推演，调整抗扰系数，并实时渲染 3D 全景图与躲避风暴的 GIF 动画。
> ```bash
> streamlit run ui/drone_ui.py
> ```
>
> ### 3. 终端极速运行模式
> 如果您仅需测试算法底层逻辑并输出静态数据图表，可直接运行主控程序：
> ```bash
> python main.py
> ```
>
> ---
>
> ## 📊 核心算法逻辑：风险敏感型 MDP
>
> 在代价函数的设计上，系统摒弃了单一的能量最优，转而采用**期望代价最小化 (Expected Cost Minimization)** 模型：
>
> $$Cost_{expected} = Energy_{normal} + P_{crash} \times Penalty_{fatal}$$
>
> - $Energy_{normal}$: 正常飞行状态下克服空气阻力与重力做功的能量积分。
> - $P_{crash}$: 由 TKE 与无人机穿越该网格的**暴露时间**共同决定的姿态失稳坠机概率。
> - $Penalty_{fatal}$: 灾难性事故的惩罚权重（可配置）。
>
> 通过上述方程，系统在“正常耗电量”与“期望坠机损失”之间进行极其精准的数学权衡，使得无人机既不会盲目悲观（绕极大的远路），也不会盲目乐观（强穿高危风暴核心）。
>
> ---
>
> ## 💡 配置与调优 (Configuration)
>
> 所有的物理引擎参数、气象极值参数以及演示设置，均集中在 `configs/config.py` 中。您可以自由调整以下关键参数以观察涌现出的智能行为：
>
> | 参数名                    | 物理意义         | 调节效果                                        |
> | :------------------------ | :--------------- | :---------------------------------------------- |
> | `enable_storms`           | 是否开启移动雷暴 | 设为 `True` 即可开启 4D 时空躲避推演            |
> | `max_ceiling`             | 最大绝对升限     | 调大允许无人机飞越喜马拉雅等极端高山            |
> | `z_weight`                | 垂直爬升权重     | 调低鼓励无人机翻越山峰，调高迫使其在山谷中穿梭  |
> | `drone_robustness_K`      | 飞行器抗扰鲁棒性 | 调低会让无人机对微小阵风极其敏感，频繁绕路      |
> | `heuristic_safety_factor` | 加权 A* 贪婪因子 | 长距离航线建议设为 `3.5` 以上，极大加快求解速度 |
>
> ---
>
> ## 🤝 贡献与二次开发 (Contribution)
>
> 本项目已实现标准化的 Gymnasium 环境封装接口思想。系统底层的 4D 物理仿真模块，未来可直接作为**多智能体强化学习（MARL）**与**具身智能无人机集群训练**的高保真数据生成基座（Data Generator）。
>
> 欢迎提交 Pull Request (PR) 或 Issue 探讨以下拓展方向：
> - [ ] 接入真实航空气象 NetCDF/GRIB 数据流
> - [ ] 基于 PPO 算法的微观姿态抗扰控制器端到端训练
> - [ ] 多无人机（Swarm）协作搜救与通信拓扑维持
>
> ---
>
> ## 📄 许可证 (License)
> 本项目基于 **MIT License** 开源。在符合相关法律法规的前提下，允许自由用于学术研究、二次开发与商业用途。
>
> *(注：README 中的演示图片及 GIF 请在本地运行程序生成后，上传至仓库的 `assets` 文件夹并替换对应链接。)*
