#!/usr/bin/env python3
"""
无人机路径规划仿真系统启动器

提供统一的入口来启动不同的系统组件：
- 传统仿真 (main.py)
- 强化学习训练 (rl_training/train_ppo.py)
- 用户界面 (ui/drone_ui.py)
- 测试套件 (pytest)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示描述"""
    print(f"🚀 {description}")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="无人机路径规划仿真系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python launcher.py sim          # 运行传统仿真
  python launcher.py rl           # 训练RL模型
  python launcher.py ui           # 启动Web界面
  python launcher.py test         # 运行测试
  python launcher.py install      # 安装依赖
        """
    )

    parser.add_argument(
        'mode',
        choices=['sim', 'rl', 'ui', 'test', 'install'],
        help='运行模式'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=500000,
        help='RL训练的总步数 (默认: 500000)'
    )

    parser.add_argument(
        '--envs',
        type=int,
        default=4,
        help='并行环境数量 (默认: 4)'
    )

    args = parser.parse_args()

    if args.mode == 'install':
        # 安装依赖
        success = run_command([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_rl_ui.txt'
        ], "安装项目依赖")

        if success:
            print("✅ 依赖安装完成")
        else:
            print("❌ 依赖安装失败")
            sys.exit(1)

    elif args.mode == 'sim':
        # 运行传统仿真
        success = run_command([
            sys.executable, 'main.py'
        ], "启动传统路径规划仿真")

    elif args.mode == 'rl':
        # 运行RL训练
        success = run_command([
            sys.executable, 'rl_training/train_ppo.py'
        ], f"启动PPO强化学习训练 (步数: {args.timesteps}, 环境数: {args.envs})")

    elif args.mode == 'ui':
        # 启动Web界面
        # 启动Web界面
        success = run_command([
            sys.executable, '-m', 'streamlit', 'run', r'C:\Users\20340\Desktop\project1\ui\drone_ui.py'
        ], "启动Web用户界面")
        
        if success:
            print("✅ Web界面已启动")
            print("📱 请在浏览器中访问显示的地址")

    elif args.mode == 'test':
        # 运行测试
        success = run_command([
            sys.executable, '-m', 'pytest', 'tests/', '-v'
        ], "运行测试套件")

    print("\n" + "="*50)
    if success:
        print("✅ 任务完成")
    else:
        print("❌ 任务失败")
        sys.exit(1)


if __name__ == "__main__":
    main()