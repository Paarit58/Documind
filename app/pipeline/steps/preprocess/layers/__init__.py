"""
Preprocessing layer modules.

This package contains the main preprocessing layers:
- Layer 1: Color & Lighting
- Layer 2: Geometric
- Layer 3: Signal-to-Noise
- Layer 4: Binarization
- Layer 5: Structural Analysis
"""

from app.pipeline.steps.preprocess.layers.binarization import apply_binarization_layer
from app.pipeline.steps.preprocess.layers.color_lighting import apply_color_lighting_layer
from app.pipeline.steps.preprocess.layers.geometric import apply_geometric_layer
from app.pipeline.steps.preprocess.layers.signal_noise import apply_signal_noise_layer
from app.pipeline.steps.preprocess.layers.structural_analysis import (
    define_kernels,
    extract_morphological_masks,
    refine_grid_masks,
    generate_intelligent_mask,
    generate_final_clean_zone_mask,
)

__all__ = [
    "apply_color_lighting_layer",
    "apply_geometric_layer",
    "apply_signal_noise_layer",
    "apply_binarization_layer",
    "define_kernels",
    "extract_morphological_masks",
    "refine_grid_masks",
    "generate_intelligent_mask",
    "generate_final_clean_zone_mask",
]

