# Desktop 启动层说明（Streamlit + pywebview）

本项目已新增“桌面化启动层”，不改原有业务逻辑与 `ui/drone_ui.py` 主界面。

## 1. 开发环境直接启动

安装依赖：

```bash
pip install -r requirements.txt
```

启动方式（任选其一）：

```bash
python desktop_launcher.py
```

```bash
python launcher.py desktop
```

Windows 双击无控制台模式：

- 直接双击 `desktop_launcher.pyw`

## 2. 启动行为

- 自动拉起本地 Streamlit 服务（无需手动 `streamlit run`）。
- 自动打开 pywebview 桌面窗口（无需手动浏览器）。
- 默认监听 `127.0.0.1:8501`，若端口占用会自动切换到可用端口。
- 运行日志写入 `logs/desktop_launcher_*.log`。

## 3. PyInstaller 打包（Windows）

一键打包：

- 双击 `build_desktop.bat`

或命令行：

```bash
python -m PyInstaller --noconfirm --clean pyinstaller_desktop.spec
```

产物路径：

- `dist/DroneDesktopLauncher/DroneDesktopLauncher.exe`

## 4. 说明

- 桌面启动器不会改动原有 `ui/drone_ui.py` 与核心算法模块。
- 若出现启动异常，可先查看 `logs/desktop_launcher_*.log` 定位问题。
