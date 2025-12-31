"""
Comb Field Removal Layer.

This layer implements the exact working script logic for removing comb fields
(form lines) from documents while preserving text.

Steps:
1. Binary conversion (adaptive threshold)
2. Raw line extraction (morphological opening)
3. Soft classification (short vs long verticals)
4. Intersection + text rejection + seriality filtering
5. Horizontal line filtering
6. Cleanup & inpainting
"""

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.utils import cv2_to_pil, image_to_base64


def apply_comb_field_removal(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    original_image: Image.Image,
    original_before_binarization: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply comb field removal using exact working script logic.
    
    This function implements the 6-step process from the working script:
    1. Binary conversion (adaptive threshold)
    2. Raw line extraction (morphological opening)
    3. Soft classification (short vs long verticals)
    4. Intersection + text rejection + seriality filtering
    5. Horizontal line filtering
    6. Cleanup & inpainting
    
    Args:
        cv_image: OpenCV image (BGR or grayscale)
        config: Preprocessing configuration
        original_image: Original PIL image for reference
        
    Returns:
        Tuple of (processed cv_image, phase_outputs_dict)
    """
    phase_outputs = {}
    original_size = cv_image.shape[:2][::-1]  # (width, height)
    
    # Get parameters from config
    k_h = config.comb_k_h
    k_w = config.comb_k_w
    max_h = config.comb_max_h
    min_serial = config.comb_min_serial
    max_stroke_width = config.comb_max_stroke_width
    min_aspect_ratio = config.comb_min_aspect_ratio
    min_long_intersections = config.comb_min_long_intersections
    
    # Convert to grayscale if needed
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
    
    # --- STEP 1: PREPROCESSING (Binary Conversion) ---
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    # Store binary output
    binary_pil = Image.fromarray(binary)
    phase_outputs["binary"] = {
        "image": image_to_base64(binary_pil),
        "size": binary_pil.size,
    }
    
    # --- STEP 2: RAW LINE EXTRACTION ---
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
    vertical_all = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horizontal_all = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    
    # Store raw line extraction outputs
    vertical_all_pil = Image.fromarray(vertical_all)
    horizontal_all_pil = Image.fromarray(horizontal_all)
    phase_outputs["raw_vertical_lines"] = {
        "image": image_to_base64(vertical_all_pil),
        "size": vertical_all_pil.size,
    }
    phase_outputs["raw_horizontal_lines"] = {
        "image": image_to_base64(horizontal_all_pil),
        "size": horizontal_all_pil.size,
    }
    
    # --- STEP 3: SOFT CLASSIFICATION (SHORT vs LONG) ---
    v_num, v_labels, v_stats, v_centroids = cv2.connectedComponentsWithStats(vertical_all)
    short_verticals = np.zeros_like(vertical_all)
    long_verticals = np.zeros_like(vertical_all)
    
    for i in range(1, v_num):
        x, y, w, h, area = v_stats[i]
        stroke_width = area / max(h, 1)
        if h < max_h:
            short_verticals[v_labels == i] = 255
        else:
            # keep long verticals ONLY if thin
            if stroke_width <= 2.5:
                long_verticals[v_labels == i] = 255
    
    # Store classification outputs
    short_verticals_pil = Image.fromarray(short_verticals)
    long_verticals_pil = Image.fromarray(long_verticals)
    phase_outputs["short_vertical_candidates"] = {
        "image": image_to_base64(short_verticals_pil),
        "size": short_verticals_pil.size,
    }
    phase_outputs["long_vertical_candidates"] = {
        "image": image_to_base64(long_verticals_pil),
        "size": long_verticals_pil.size,
    }
    
    # Combine both for analysis
    vertical_candidates = cv2.add(short_verticals, long_verticals)
    
    # --- STEP 4: INTERSECTION + TEXT REJECTION + SERIALITY ---
    v_num_f, v_labels_f, v_stats_f, v_centroids_f = cv2.connectedComponentsWithStats(vertical_candidates)
    v_to_remove = np.zeros_like(binary)
    h_dilated = cv2.dilate(horizontal_all, np.ones((3, 3), np.uint8))
    rows = {}
    tolerance = 10
    intersection_visual = np.zeros_like(binary)
    rejected_as_text = np.zeros_like(binary)
    accepted_long_grid = np.zeros_like(binary)
    
    for i in range(1, v_num_f):
        x, y, w, h, area = v_stats_f[i]
        cx, cy = v_centroids_f[i]
        stroke_width = area / max(h, 1)
        aspect_ratio = h / max(w, 1)
        
        # --- TEXT REJECTION ---
        if stroke_width > max_stroke_width or aspect_ratio < min_aspect_ratio:
            rejected_as_text[v_labels_f == i] = 255
            continue
        
        component_mask = (v_labels_f == i).astype("uint8") * 255
        intersection = cv2.bitwise_and(component_mask, h_dilated)
        if not intersection.any():
            continue
        
        intersection_visual = cv2.add(intersection_visual, intersection)
        
        # --- LONG VERTICAL SPECIAL HANDLING ---
        if h >= max_h:
            intersection_count = cv2.countNonZero(intersection) // max(int(stroke_width), 1)
            if intersection_count < min_long_intersections:
                continue  # long but not grid-like
            accepted_long_grid[v_labels_f == i] = 255
        
        # --- ROW GROUPING ---
        found_row = False
        for r_y in rows.keys():
            if abs(cy - r_y) < tolerance:
                rows[r_y].append(i)
                found_row = True
                break
        if not found_row:
            rows[cy] = [i]
    
    # Store intersection analysis outputs
    intersection_visual_pil = Image.fromarray(intersection_visual)
    rejected_as_text_pil = Image.fromarray(rejected_as_text)
    accepted_long_grid_pil = Image.fromarray(accepted_long_grid)
    phase_outputs["intersection_visualization"] = {
        "image": image_to_base64(intersection_visual_pil),
        "size": intersection_visual_pil.size,
    }
    phase_outputs["rejected_as_text"] = {
        "image": image_to_base64(rejected_as_text_pil),
        "size": rejected_as_text_pil.size,
    }
    phase_outputs["accepted_long_grid"] = {
        "image": image_to_base64(accepted_long_grid_pil),
        "size": accepted_long_grid_pil.size,
    }
    
    # --- STEP 4d: SERIALITY FILTER ---
    for r_y, indices in rows.items():
        if len(indices) >= min_serial:
            for idx in indices:
                v_to_remove[v_labels_f == idx] = 255
    
    # Store final vertical mask
    v_to_remove_pil = Image.fromarray(v_to_remove)
    phase_outputs["final_vertical_mask"] = {
        "image": image_to_base64(v_to_remove_pil),
        "size": v_to_remove_pil.size,
    }
    
    # --- STEP 5: HORIZONTAL LINE FILTERING ---
    h_to_remove = np.zeros_like(binary)
    h_num, h_labels, _, _ = cv2.connectedComponentsWithStats(horizontal_all)
    v_to_remove_dilated = cv2.dilate(v_to_remove, np.ones((3, 3), np.uint8))
    
    for i in range(1, h_num):
        h_mask = (h_labels == i).astype("uint8") * 255
        if cv2.bitwise_and(h_mask, v_to_remove_dilated).any():
            h_to_remove = cv2.add(h_to_remove, h_mask)
    
    # Store final horizontal mask
    h_to_remove_pil = Image.fromarray(h_to_remove)
    phase_outputs["final_horizontal_mask"] = {
        "image": image_to_base64(h_to_remove_pil),
        "size": h_to_remove_pil.size,
    }
    
    # --- STEP 6: CLEANUP & INPAINTING ---
    full_mask = cv2.add(v_to_remove, h_to_remove)
    full_mask_dilated = cv2.dilate(full_mask, np.ones((3, 3), np.uint8))
    
    # Use original image before binarization for inpainting (like working script)
    # If original_before_binarization is provided, use it; otherwise use current cv_image
    if original_before_binarization is not None:
        # Ensure it's BGR format
        if len(original_before_binarization.shape) == 2:
            img_for_inpaint = cv2.cvtColor(original_before_binarization, cv2.COLOR_GRAY2BGR)
        else:
            img_for_inpaint = original_before_binarization.copy()
    else:
        # Fallback: use current cv_image
        if len(cv_image.shape) == 2:
            img_for_inpaint = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        else:
            img_for_inpaint = cv_image.copy()
    
    final_result = cv2.inpaint(img_for_inpaint, full_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    # Store final result
    final_result_pil = cv2_to_pil(final_result)
    phase_outputs["final_result"] = {
        "image": image_to_base64(final_result_pil),
        "size": final_result_pil.size,
    }
    
    # Store full mask visualization
    full_mask_pil = Image.fromarray(full_mask_dilated)
    phase_outputs["full_mask"] = {
        "image": image_to_base64(full_mask_pil),
        "size": full_mask_pil.size,
    }
    
    # Metadata
    phase_outputs["metadata"] = {
        "original_size": original_size,
        "final_size": final_result.shape[:2][::-1],
        "parameters": {
            "k_h": k_h,
            "k_w": k_w,
            "max_h": max_h,
            "min_serial": min_serial,
            "max_stroke_width": max_stroke_width,
            "min_aspect_ratio": min_aspect_ratio,
            "min_long_intersections": min_long_intersections,
        },
        "statistics": {
            "vertical_lines_detected": int(v_num - 1),
            "horizontal_lines_detected": int(h_num - 1),
            "rows_with_seriality": len([r for r in rows.values() if len(r) >= min_serial]),
            "total_rows": len(rows),
        },
    }
    
    return final_result, phase_outputs

