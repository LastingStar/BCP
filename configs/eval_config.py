# --- START OF FILE configs/eval_config.py ---
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class MethodConfig:
    """定义一个测试方法的配置"""
    name: str
    kind: str  # "teacher" 或 "rl"
    model_path: Optional[Path] = None

@dataclass
class EvalConfig:
    """批量评估测试集的配置"""
    test_seeds: List[int]
    curriculum_stage: int = 2
    include_ablation: bool = False
    
@dataclass
class RenderConfig:
    """案例渲染图纸的配置"""
    output_dir: str = "results/cases"
    make_gif: bool = True
    make_3d: bool = True
