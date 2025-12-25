"""
Layer 5: Structural Analysis
- Phase 1: Kernel Definition Step
- Phase 2: Morphological Extraction Step
- Phase 3: Grid Intersection & Refinement Step
- Phase 4: Intelligent Mask Generation with Text Preservation

This module defines kernels and performs morphological operations
to detect form lines (boxes and grids) while preserving text.
"""

import cv2
import numpy as np
from PIL import Image

from app.core.config import PreprocessingConfig
from app.pipeline.steps.preprocess.utils import image_to_base64


def define_kernels(
    cv_image: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Phase 1: Define horizontal and vertical kernels for structural analysis.
    
    Kernels are dynamically scaled based on image dimensions:
    - Horizontal kernel: width = 1/N of image width, height = 1
    - Vertical kernel: height = 1/N of image height, width = 1
    
    This dynamic scaling prevents:
    - Too short kernels: mistaking letter parts (like 'E' top bar) for form lines
    - Too long kernels: missing small vertical ticks in comb fields
    
    Args:
        cv_image: OpenCV image (BGR or grayscale)
        config: Preprocessing configuration
        
    Returns:
        Tuple of (horizontal_kernel, vertical_kernel, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Get image dimensions
    if len(cv_image.shape) == 3:
        h, w = cv_image.shape[:2]
    else:
        h, w = cv_image.shape
    
    # Calculate kernel dimensions using dynamic scaling
    # Horizontal kernel width = 1/N of image width
    kernel_width = max(
        config.structural_kernel_min_size,
        min(
            config.structural_kernel_max_size,
            int(w / config.structural_kernel_ratio)
        )
    )
    
    # Vertical kernel height = 1/N of image height
    kernel_height = max(
        config.structural_kernel_min_size,
        min(
            config.structural_kernel_max_size,
            int(h / config.structural_kernel_ratio)
        )
    )
    
    # Ensure kernels are odd (required for some morphological operations)
    if kernel_width % 2 == 0:
        kernel_width += 1
    if kernel_height % 2 == 0:
        kernel_height += 1
    
    # Define kernels
    # Horizontal kernel: wide and short (detects horizontal lines)
    horizontal_kernel = np.ones((1, kernel_width), dtype=np.uint8)
    
    # Vertical kernel: tall and narrow (detects vertical lines)
    vertical_kernel = np.ones((kernel_height, 1), dtype=np.uint8)
    
    # Store kernel information for debugging/visualization
    phase_outputs["kernel_info"] = {
        "image_width": int(w),
        "image_height": int(h),
        "kernel_ratio": float(config.structural_kernel_ratio),
        "horizontal_kernel_size": (1, kernel_width),
        "vertical_kernel_size": (kernel_height, 1),
        "horizontal_kernel_width": int(kernel_width),
        "vertical_kernel_height": int(kernel_height),
    }
    
    # Create visualization of kernels (for debugging/testing)
    # Visualize horizontal kernel as a white line
    h_kernel_vis = np.zeros((50, kernel_width * 10), dtype=np.uint8)
    h_kernel_vis[20:21, :] = 255
    h_kernel_vis_pil = Image.fromarray(h_kernel_vis)
    phase_outputs["horizontal_kernel_visualization"] = image_to_base64(h_kernel_vis_pil)
    
    # Visualize vertical kernel as a white line
    v_kernel_vis = np.zeros((kernel_height * 10, 50), dtype=np.uint8)
    v_kernel_vis[:, 20:21] = 255
    v_kernel_vis_pil = Image.fromarray(v_kernel_vis)
    phase_outputs["vertical_kernel_visualization"] = image_to_base64(v_kernel_vis_pil)
    
    return horizontal_kernel, vertical_kernel, phase_outputs


def extract_morphological_masks(
    cv_image: np.ndarray,
    horizontal_kernel: np.ndarray,
    vertical_kernel: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Phase 2: Extract horizontal and vertical form lines using morphological operations.
    
    This phase:
    1. Converts image to binary (if not already)
    2. Applies morphological opening (erosion + dilation) with horizontal kernel
    3. Applies morphological opening (erosion + dilation) with vertical kernel
    4. Produces two separate masks: horizontal_lines_mask and vertical_lines_mask
    
    Morphological Opening:
    - Erosion: "Eats away" at the image. Small shapes (letters) disappear, 
               long lines (form lines) survive if they're longer than the kernel
    - Dilation: Restores the survived lines to their original thickness
    
    Args:
        cv_image: OpenCV image (BGR or grayscale, should be binarized)
        horizontal_kernel: Horizontal kernel from Phase 1
        vertical_kernel: Vertical kernel from Phase 1
        config: Preprocessing configuration
        
    Returns:
        Tuple of (horizontal_lines_mask, vertical_lines_mask, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Convert to grayscale if needed
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image.copy()
    
    # Ensure binary image (invert if needed - form lines are typically dark on light background)
    # If image is already binary, use as-is, otherwise threshold
    if len(np.unique(gray)) > 2:
        # Not binary, apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Already binary, invert so lines are white (255) on black (0) background
        binary = cv2.bitwise_not(gray) if np.mean(gray) > 127 else gray
    
    # Store original binary for visualization
    binary_pil = Image.fromarray(binary)
    phase_outputs["input_binary"] = {
        "image": image_to_base64(binary_pil),
        "size": binary_pil.size,
    }
    
    # Morphological Opening: Erosion followed by Dilation
    # This removes small objects (letters) and keeps long lines (form structure)
    
    # Horizontal lines extraction
    # Step 1: Erosion with horizontal kernel (removes everything shorter than kernel width)
    horizontal_eroded = cv2.erode(binary, horizontal_kernel, iterations=1)
    
    # Step 2: Dilation with horizontal kernel (restores survived lines to original thickness)
    horizontal_lines_mask = cv2.dilate(horizontal_eroded, horizontal_kernel, iterations=1)
    
    # Vertical lines extraction
    # Step 1: Erosion with vertical kernel (removes everything shorter than kernel height)
    vertical_eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    
    # Step 2: Dilation with vertical kernel (restores survived lines to original thickness)
    vertical_lines_mask = cv2.dilate(vertical_eroded, vertical_kernel, iterations=1)
    
    # Store intermediate erosion results for visualization
    h_eroded_pil = Image.fromarray(horizontal_eroded)
    v_eroded_pil = Image.fromarray(vertical_eroded)
    
    phase_outputs["horizontal_erosion"] = {
        "image": image_to_base64(h_eroded_pil),
        "size": h_eroded_pil.size,
    }
    phase_outputs["vertical_erosion"] = {
        "image": image_to_base64(v_eroded_pil),
        "size": v_eroded_pil.size,
    }
    
    # Store final masks
    h_mask_pil = Image.fromarray(horizontal_lines_mask)
    v_mask_pil = Image.fromarray(vertical_lines_mask)
    
    phase_outputs["horizontal_lines_mask"] = {
        "image": image_to_base64(h_mask_pil),
        "size": h_mask_pil.size,
    }
    phase_outputs["vertical_lines_mask"] = {
        "image": image_to_base64(v_mask_pil),
        "size": v_mask_pil.size,
    }
    
    # Calculate statistics
    h_pixels = np.sum(horizontal_lines_mask > 0)
    v_pixels = np.sum(vertical_lines_mask > 0)
    total_pixels = horizontal_lines_mask.size
    
    phase_outputs["statistics"] = {
        "horizontal_lines_pixels": int(h_pixels),
        "vertical_lines_pixels": int(v_pixels),
        "horizontal_lines_percentage": float((h_pixels / total_pixels) * 100),
        "vertical_lines_percentage": float((v_pixels / total_pixels) * 100),
    }
    
    return horizontal_lines_mask, vertical_lines_mask, phase_outputs


def _classify_intersection_type(
    y: int, x: int, h_mask: np.ndarray, v_mask: np.ndarray, radius: int = 3
) -> str:
    """
    Classify intersection type based on nearby lines.
    
    Args:
        y, x: Intersection coordinates
        h_mask: Horizontal mask
        v_mask: Vertical mask
        radius: Radius to check for lines
        
    Returns:
        Intersection type: 'corner', 't_junction', 'cross', 'plus'
    """
    h, w = h_mask.shape
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    
    # Count horizontal lines (check left and right)
    h_left = np.sum(h_mask[y, max(0, x - radius):x] > 0)
    h_right = np.sum(h_mask[y, x:min(w, x + radius + 1)] > 0)
    h_count = 1 if (h_left > 0 or h_right > 0) else 0
    
    # Count vertical lines (check top and bottom)
    v_top = np.sum(v_mask[max(0, y - radius):y, x] > 0)
    v_bottom = np.sum(v_mask[y:min(h, y + radius + 1), x] > 0)
    v_count = 1 if (v_top > 0 or v_bottom > 0) else 0
    
    total_lines = h_count + v_count
    
    if total_lines == 2:
        return "corner"
    elif total_lines == 3:
        return "t_junction"
    elif total_lines == 4:
        return "cross"
    else:
        return "plus"


def _detect_comb_fields_pattern_based(
    h_mask: np.ndarray,
    v_mask: np.ndarray,
    config: PreprocessingConfig,
) -> list[dict]:
    """
    Enhanced comb field detection using pattern analysis.
    
    This method directly analyzes the vertical mask to find patterns of
    multiple vertical lines between horizontal line pairs, making it more
    robust than intersection-based detection.
    
    Algorithm:
    1. Find horizontal line segments (connected components)
    2. Group nearby horizontal lines into pairs (top/bottom)
    3. For each pair, analyze vertical mask region:
       - Count vertical lines in the region
       - Check spacing consistency
       - Validate vertical coverage
    4. Return validated comb fields
    
    Args:
        h_mask: Horizontal lines mask
        v_mask: Vertical lines mask
        config: Preprocessing configuration
        
    Returns:
        List of comb field dictionaries with coordinates
    """
    comb_fields = []
    h, w = h_mask.shape
    
    # Early return if masks are empty
    if np.sum(h_mask > 0) == 0 or np.sum(v_mask > 0) == 0:
        return []
    
    # Step 1: Find horizontal line segments using connected components
    h_num_labels, h_labels, h_stats, h_centroids = cv2.connectedComponentsWithStats(
        h_mask, connectivity=8
    )
    
    # Extract horizontal line segments with their y-coordinates and x-ranges
    horizontal_segments = []
    for label_id in range(1, h_num_labels):  # Skip background (label 0)
        y = int(h_centroids[label_id][1])  # Centroid y-coordinate
        x = h_stats[label_id, 0]  # Left
        width = h_stats[label_id, 2]  # Width
        x_min = x
        x_max = x + width
        
        # Get actual y-coordinate from the line (use top of bounding box)
        y_coord = h_stats[label_id, 1]  # Top
        
        horizontal_segments.append({
            "y": int(y_coord),
            "x_min": int(x_min),
            "x_max": int(x_max),
            "width": int(width),
            "label": label_id,
        })
    
    # Sort by y-coordinate
    horizontal_segments.sort(key=lambda s: s["y"])
    
    # Early return if not enough horizontal segments
    if len(horizontal_segments) < 2:
        return []
    
    # Step 2: Find pairs of horizontal lines that might form comb fields
    # Look for pairs with similar x-ranges and reasonable vertical distance
    max_vertical_distance = int(h * 0.15)  # Max 15% of image height
    min_vertical_distance = 5  # Minimum distance between horizontals
    
    for i in range(len(horizontal_segments)):
        top_seg = horizontal_segments[i]
        top_y = top_seg["y"]
        top_x_min = top_seg["x_min"]
        top_x_max = top_seg["x_max"]
        
        # Look for bottom horizontal line
        for j in range(i + 1, len(horizontal_segments)):
            bottom_seg = horizontal_segments[j]
            bottom_y = bottom_seg["y"]
            bottom_x_min = bottom_seg["x_min"]
            bottom_x_max = bottom_seg["x_max"]
            
            # Check vertical distance
            vertical_dist = bottom_y - top_y
            if vertical_dist < min_vertical_distance:
                continue
            if vertical_dist > max_vertical_distance:
                break  # Too far, skip remaining segments
            
            # Check x-range overlap (horizontal lines should have similar x-ranges)
            overlap_x_min = max(top_x_min, bottom_x_min)
            overlap_x_max = min(top_x_max, bottom_x_max)
            overlap_width = overlap_x_max - overlap_x_min
            
            if overlap_width < 20:  # Minimum width for comb field
                continue
            
            # Step 3: Analyze vertical mask region between these two horizontals
            # Expand region slightly for tolerance
            tolerance = config.structural_comb_field_vertical_tolerance
            region_top_y = max(0, top_y - tolerance)
            region_bottom_y = min(h, bottom_y + tolerance + 1)
            region_left_x = max(0, overlap_x_min - tolerance)
            region_right_x = min(w, overlap_x_max + tolerance + 1)
            
            if region_bottom_y <= region_top_y or region_right_x <= region_left_x:
                continue
            
            # Extract vertical mask region
            region_v_mask = v_mask[region_top_y:region_bottom_y, region_left_x:region_right_x]
            
            if np.sum(region_v_mask > 0) == 0:
                continue  # No vertical lines in region
            
            # Count vertical lines using connected components
            v_num_labels, v_labels, v_stats, v_centroids = cv2.connectedComponentsWithStats(
                region_v_mask, connectivity=8
            )
            
            # Filter vertical lines by height (must cover sufficient portion of region height)
            region_height = region_bottom_y - region_top_y
            min_vertical_height = int(region_height * config.structural_comb_field_min_vertical_coverage)
            
            vertical_lines = []
            for v_label_id in range(1, v_num_labels):
                v_height = v_stats[v_label_id, 3]  # Height
                if v_height >= min_vertical_height:
                    v_x = int(v_centroids[v_label_id][0]) + region_left_x  # Global x-coordinate
                    vertical_lines.append({
                        "x": v_x,
                        "height": v_height,
                        "label": v_label_id,
                    })
            
            # Check if we have enough vertical lines
            if len(vertical_lines) < config.structural_comb_field_min_lines:
                continue
            
            # Step 4: Validate spacing consistency (comb fields have regular spacing)
            # Note: Spacing validation is lenient to handle real-world variations
            vertical_lines.sort(key=lambda l: l["x"])
            if len(vertical_lines) >= 2:
                spacings = []
                for k in range(len(vertical_lines) - 1):
                    spacing = vertical_lines[k + 1]["x"] - vertical_lines[k]["x"]
                    if spacing > 0:  # Only consider positive spacings
                        spacings.append(spacing)
                
                # Check if spacings are relatively consistent (within 70% variation for leniency)
                if spacings and len(spacings) > 0:
                    avg_spacing = np.mean(spacings)
                    max_allowed_spacing = w * config.structural_comb_field_max_line_spacing_ratio
                    
                    # Skip spacing validation if average spacing is very small (tight comb fields)
                    # or if we have many lines (likely a comb field regardless of spacing)
                    if avg_spacing > 5 and len(vertical_lines) < 20:
                        # Validate: most spacings should be similar and not too large
                        consistent_spacings = [s for s in spacings if abs(s - avg_spacing) < avg_spacing * 0.7 and s < max_allowed_spacing]
                        
                        if len(consistent_spacings) < len(spacings) * 0.5:  # At least 50% consistent (lenient)
                            continue  # Spacing too irregular, probably not a comb field
            
            # This is a valid comb field
            x_coordinates = [line["x"] for line in vertical_lines]
            left_x = min(x_coordinates)
            right_x = max(x_coordinates)
            
            comb_fields.append({
                "top_y": int(top_y),
                "bottom_y": int(bottom_y),
                "left_x": int(left_x),
                "right_x": int(right_x),
                "vertical_lines": len(vertical_lines),
                "x_coordinates": x_coordinates,
            })
    
    return comb_fields


def _detect_comb_fields_intersection_based(
    intersection_points: np.ndarray,
    h_mask: np.ndarray,
    v_mask: np.ndarray,
    min_lines: int,
    cluster_threshold: int,
) -> list[dict]:
    """
    Detect comb fields from intersection points (fallback method).
    
    A comb field is a series of vertical lines between two horizontal lines.
    This is the original intersection-based detection method.
    
    Args:
        intersection_points: Array of (y, x) intersection coordinates
        h_mask: Horizontal mask
        v_mask: Vertical mask
        min_lines: Minimum number of vertical lines for comb field
        cluster_threshold: Distance threshold for clustering
        
    Returns:
        List of comb field dictionaries with coordinates
    """
    if len(intersection_points) == 0:
        return []
    
    comb_fields = []
    h, w = h_mask.shape
    
    # Group intersections by y-coordinate (horizontal lines)
    y_coords = {}
    for point in intersection_points:
        y, x = int(point[0]), int(point[1])
        if y not in y_coords:
            y_coords[y] = []
        y_coords[y].append(x)
    
    # Find pairs of horizontal lines that might form comb fields
    sorted_y = sorted(y_coords.keys())
    
    for i in range(len(sorted_y) - 1):
        y1 = sorted_y[i]
        y2 = sorted_y[i + 1]
        
        # Check if there are vertical lines between these two horizontal lines
        x_coords1 = sorted(y_coords[y1])
        x_coords2 = sorted(y_coords[y2])
        
        # Find common x-coordinates (vertical lines that intersect both horizontals)
        common_x = sorted(set(x_coords1) & set(x_coords2))
        
        if len(common_x) >= min_lines:
            # This is a comb field
            x_min = min(common_x)
            x_max = max(common_x)
            
            comb_fields.append({
                "top_y": int(y1),
                "bottom_y": int(y2),
                "left_x": int(x_min),
                "right_x": int(x_max),
                "vertical_lines": len(common_x),
                "x_coordinates": [int(x) for x in common_x],
            })
    
    return comb_fields


def _detect_comb_fields(
    intersection_points: np.ndarray,
    h_mask: np.ndarray,
    v_mask: np.ndarray,
    config: PreprocessingConfig,
) -> list[dict]:
    """
    Detect comb fields using dual detection approach.
    
    Combines pattern-based detection (primary) with intersection-based detection (fallback)
    to ensure maximum coverage of comb field structures.
    
    Args:
        intersection_points: Array of (y, x) intersection coordinates
        h_mask: Horizontal mask
        v_mask: Vertical mask
        config: Preprocessing configuration
        
    Returns:
        List of comb field dictionaries with coordinates (merged from both methods)
    """
    comb_fields = []
    
    # Primary: Pattern-based detection (more robust)
    if config.structural_comb_field_pattern_detection:
        pattern_fields = _detect_comb_fields_pattern_based(h_mask, v_mask, config)
        comb_fields.extend(pattern_fields)
    
    # Fallback: Intersection-based detection (catches edge cases)
    intersection_fields = _detect_comb_fields_intersection_based(
        intersection_points,
        h_mask,
        v_mask,
        config.structural_comb_field_min_lines,
        config.structural_intersection_cluster_threshold,
    )
    
    # Merge results, removing duplicates
    # Two comb fields are considered duplicates if they overlap significantly
    all_fields = comb_fields + intersection_fields
    
    if not all_fields:
        return []
    
    # Remove duplicates based on overlap
    merged_fields = []
    for field in all_fields:
        is_duplicate = False
        for existing in merged_fields:
            # Check overlap: if bounding boxes overlap by >50%, consider duplicate
            overlap_top = max(field["top_y"], existing["top_y"])
            overlap_bottom = min(field["bottom_y"], existing["bottom_y"])
            overlap_left = max(field["left_x"], existing["left_x"])
            overlap_right = min(field["right_x"], existing["right_x"])
            
            if overlap_bottom > overlap_top and overlap_right > overlap_left:
                overlap_area = (overlap_bottom - overlap_top) * (overlap_right - overlap_left)
                field_area = (field["bottom_y"] - field["top_y"]) * (field["right_x"] - field["left_x"])
                existing_area = (existing["bottom_y"] - existing["top_y"]) * (existing["right_x"] - existing["left_x"])
                
                overlap_ratio_field = overlap_area / field_area if field_area > 0 else 0
                overlap_ratio_existing = overlap_area / existing_area if existing_area > 0 else 0
                
                if overlap_ratio_field > 0.5 or overlap_ratio_existing > 0.5:
                    # Merge: keep the one with more vertical lines or larger area
                    if field["vertical_lines"] > existing["vertical_lines"] or field_area > existing_area:
                        merged_fields.remove(existing)
                        merged_fields.append(field)
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            merged_fields.append(field)
    
    return merged_fields


def _filter_segments_with_context(
    mask: np.ndarray,
    min_length: int,
    intersection_mask: np.ndarray,
    is_horizontal: bool,
    context_radius: int,
) -> np.ndarray:
    """
    Filter segments with context awareness.
    
    Keep short segments if they're near intersections or connect to longer lines.
    
    Args:
        mask: Input mask (horizontal or vertical)
        min_length: Minimum length threshold
        intersection_mask: Mask of intersection points
        is_horizontal: True for horizontal lines, False for vertical
        context_radius: Radius for context checking
        
    Returns:
        Filtered mask
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    filtered_mask = np.zeros_like(mask)
    
    # Dilate intersection mask for context checking
    context_kernel = np.ones((context_radius * 2 + 1, context_radius * 2 + 1), dtype=np.uint8)
    dilated_intersections = cv2.dilate(intersection_mask, context_kernel, iterations=1)
    
    for label_id in range(1, num_labels):
        width = stats[label_id, 2]
        height = stats[label_id, 3]
        x = stats[label_id, 0]
        y = stats[label_id, 1]
        
        if is_horizontal:
            length = width
            aspect_ratio = width / height if height > 0 else 0
        else:
            length = height
            aspect_ratio = height / width if width > 0 else 0
        
        # Check if segment meets length requirement
        meets_length = length >= min_length and aspect_ratio > 2.0
        
        # Check if segment is near an intersection (context-aware)
        segment_center_y = y + height // 2
        segment_center_x = x + width // 2
        near_intersection = dilated_intersections[segment_center_y, segment_center_x] > 0
        
        # Keep if meets length OR is near intersection
        if meets_length or near_intersection:
            filtered_mask[labels == label_id] = 255
    
    return filtered_mask


def _identify_boxes_from_intersections(
    intersection_points: np.ndarray,
    h_mask: np.ndarray,
    v_mask: np.ndarray,
) -> list[dict]:
    """
    Identify boxes/fields from intersection points.
    
    Args:
        intersection_points: Array of (y, x) intersection coordinates
        h_mask: Horizontal mask
        v_mask: Vertical mask
        
    Returns:
        List of detected boxes with coordinates
    """
    if len(intersection_points) == 0:
        return []
    
    boxes = []
    h, w = h_mask.shape
    
    # Group intersections by y-coordinate
    y_coords = {}
    for point in intersection_points:
        y, x = int(point[0]), int(point[1])
        if y not in y_coords:
            y_coords[y] = []
        y_coords[y].append(x)
    
    # Find rectangular regions
    sorted_y = sorted(y_coords.keys())
    
    for i in range(len(sorted_y) - 1):
        y1 = sorted_y[i]
        y2 = sorted_y[i + 1]
        
        x_coords1 = sorted(y_coords[y1])
        x_coords2 = sorted(y_coords[y2])
        
        # Find aligned x-coordinates (potential box corners)
        for j in range(len(x_coords1) - 1):
            x1 = x_coords1[j]
            x2 = x_coords1[j + 1]
            
            # Check if this forms a box (has corresponding corners)
            if x1 in x_coords2 and x2 in x_coords2:
                boxes.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                })
    
    return boxes


def refine_grid_masks(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    cv_image: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Phase 3: Refine grid masks by finding intersections and filtering short segments.
    
    This phase:
    1. Creates Junction Map (wireframe) by combining horizontal and vertical masks
    2. Finds intersection points (grid corners) with enhanced analysis
    3. Detects comb fields from intersection patterns
    4. Filters short segments with context awareness
    5. Identifies boxes/fields from intersection points
    
    Args:
        horizontal_mask: Horizontal lines mask from Phase 2
        vertical_mask: Vertical lines mask from Phase 2
        cv_image: Original OpenCV image for dimension reference
        config: Preprocessing configuration
        
    Returns:
        Tuple of (refined_horizontal_mask, refined_vertical_mask, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Get image dimensions
    if len(cv_image.shape) == 3:
        h, w = cv_image.shape[:2]
    else:
        h, w = cv_image.shape
    
    image_area = h * w
    
    # Calculate minimum line length
    min_h_length = max(1, int(w * config.structural_min_line_length_ratio))
    min_v_length = max(1, int(h * config.structural_min_line_length_ratio))
    
    # Step 0: Create Junction Map (Wireframe)
    junction_map = cv2.bitwise_or(horizontal_mask, vertical_mask)
    
    # Create color-coded visualization (always create for use in other visualizations)
    wireframe_vis = np.zeros((h, w, 3), dtype=np.uint8)
    wireframe_vis[horizontal_mask > 0] = [255, 0, 0]  # Blue for horizontal
    wireframe_vis[vertical_mask > 0] = [0, 0, 255]   # Red for vertical
    
    if config.structural_enable_junction_map:
        # Calculate wireframe statistics
        wireframe_pixels = np.sum(junction_map > 0)
        wireframe_coverage = (wireframe_pixels / image_area) * 100
        
        phase_outputs["junction_map"] = {
            "image": image_to_base64(Image.fromarray(cv2.cvtColor(wireframe_vis, cv2.COLOR_BGR2RGB))),
            "size": (w, h),
            "pixels": int(wireframe_pixels),
            "coverage_percentage": float(wireframe_coverage),
        }
    
    # Step 1: Find intersection points (grid corners)
    intersection_mask = cv2.bitwise_and(horizontal_mask, vertical_mask)
    
    # Extract intersection coordinates
    intersection_points = np.column_stack(np.where(intersection_mask > 0))
    
    # Dilate intersections slightly to mark grid corners
    intersection_kernel = np.ones(
        (config.structural_intersection_threshold, config.structural_intersection_threshold),
        dtype=np.uint8
    )
    dilated_intersections = cv2.dilate(intersection_mask, intersection_kernel, iterations=1)
    
    # Classify intersection types
    intersection_types = {}
    for point in intersection_points:
        y, x = int(point[0]), int(point[1])
        int_type = _classify_intersection_type(y, x, horizontal_mask, vertical_mask)
        if int_type not in intersection_types:
            intersection_types[int_type] = 0
        intersection_types[int_type] += 1
    
    # Enhanced intersection visualization
    intersection_vis = np.zeros((h, w, 3), dtype=np.uint8)
    intersection_vis[horizontal_mask > 0] = [255, 0, 0]  # Blue for horizontal
    intersection_vis[vertical_mask > 0] = [0, 0, 255]   # Red for vertical
    intersection_vis[dilated_intersections > 0] = [0, 255, 0]  # Green for intersections
    
    intersection_vis_pil = Image.fromarray(cv2.cvtColor(intersection_vis, cv2.COLOR_BGR2RGB))
    phase_outputs["intersections"] = {
        "image": image_to_base64(intersection_vis_pil),
        "size": intersection_vis_pil.size,
        "count": int(len(intersection_points)),
        "types": intersection_types,
    }
    
    # Step 1.5: Detect Comb Fields
    comb_fields = []
    if config.structural_enable_comb_field_detection:
        comb_fields = _detect_comb_fields(
            intersection_points,
            horizontal_mask,
            vertical_mask,
            config,
        )
        
        # Visualize comb fields
        if comb_fields:
            comb_vis = wireframe_vis.copy()
            for field in comb_fields:
                # Draw bounding box around comb field
                cv2.rectangle(
                    comb_vis,
                    (field["left_x"], field["top_y"]),
                    (field["right_x"], field["bottom_y"]),
                    (255, 255, 0),  # Yellow for comb fields
                    2
                )
            comb_vis_pil = Image.fromarray(cv2.cvtColor(comb_vis, cv2.COLOR_BGR2RGB))
            phase_outputs["comb_fields"] = {
                "image": image_to_base64(comb_vis_pil),
                "size": comb_vis_pil.size,
                "count": len(comb_fields),
                "fields": comb_fields,
            }
    
    # Step 2: Enhanced Length Filtering with Context Awareness
    refined_h_mask = _filter_segments_with_context(
        horizontal_mask,
        min_h_length,
        intersection_mask,
        is_horizontal=True,
        context_radius=config.structural_segment_context_radius,
    )
    
    refined_v_mask = _filter_segments_with_context(
        vertical_mask,
        min_v_length,
        intersection_mask,
        is_horizontal=False,
        context_radius=config.structural_segment_context_radius,
    )
    
    # Calculate filtering statistics
    h_num_labels_before, _, _, _ = cv2.connectedComponentsWithStats(horizontal_mask, connectivity=8)
    h_num_labels_after, _, _, _ = cv2.connectedComponentsWithStats(refined_h_mask, connectivity=8)
    v_num_labels_before, _, _, _ = cv2.connectedComponentsWithStats(vertical_mask, connectivity=8)
    v_num_labels_after, _, _, _ = cv2.connectedComponentsWithStats(refined_v_mask, connectivity=8)
    
    filtered_h_count = (h_num_labels_before - 1) - (h_num_labels_after - 1)
    filtered_v_count = (v_num_labels_before - 1) - (v_num_labels_after - 1)
    
    # Step 3: Connect broken segments near intersections
    connect_kernel = np.ones((3, 3), dtype=np.uint8)
    refined_h_mask = cv2.dilate(refined_h_mask, connect_kernel, iterations=1)
    refined_v_mask = cv2.dilate(refined_v_mask, connect_kernel, iterations=1)
    
    # Erode back to original thickness
    refined_h_mask = cv2.erode(refined_h_mask, connect_kernel, iterations=1)
    refined_v_mask = cv2.erode(refined_v_mask, connect_kernel, iterations=1)
    
    # Step 4: Identify Boxes/Fields from Intersections
    detected_boxes = _identify_boxes_from_intersections(
        intersection_points, refined_h_mask, refined_v_mask
    )
    
    # Visualize detected boxes
    if detected_boxes:
        box_vis = wireframe_vis.copy()
        for box in detected_boxes:
            cv2.rectangle(
                box_vis,
                (box["x1"], box["y1"]),
                (box["x2"], box["y2"]),
                (0, 255, 255),  # Cyan for boxes
                2
            )
        box_vis_pil = Image.fromarray(cv2.cvtColor(box_vis, cv2.COLOR_BGR2RGB))
        phase_outputs["detected_boxes"] = {
            "image": image_to_base64(box_vis_pil),
            "size": box_vis_pil.size,
            "count": len(detected_boxes),
            "boxes": detected_boxes,
        }
    
    # Store refined masks for visualization
    refined_h_pil = Image.fromarray(refined_h_mask)
    refined_v_pil = Image.fromarray(refined_v_mask)
    
    phase_outputs["refined_horizontal_mask"] = {
        "image": image_to_base64(refined_h_pil),
        "size": refined_h_pil.size,
    }
    phase_outputs["refined_vertical_mask"] = {
        "image": image_to_base64(refined_v_pil),
        "size": refined_v_pil.size,
    }
    
    # Enhanced Statistics
    h_pixels_before = np.sum(horizontal_mask > 0)
    h_pixels_after = np.sum(refined_h_mask > 0)
    v_pixels_before = np.sum(vertical_mask > 0)
    v_pixels_after = np.sum(refined_v_mask > 0)
    
    phase_outputs["statistics"] = {
        "horizontal_pixels_before": int(h_pixels_before),
        "horizontal_pixels_after": int(h_pixels_after),
        "horizontal_filtered_segments": int(filtered_h_count),
        "vertical_pixels_before": int(v_pixels_before),
        "vertical_pixels_after": int(v_pixels_after),
        "vertical_filtered_segments": int(filtered_v_count),
        "min_horizontal_length": int(min_h_length),
        "min_vertical_length": int(min_v_length),
        "intersection_count": int(len(intersection_points)),
        "comb_fields_count": len(comb_fields),
        "detected_boxes_count": len(detected_boxes),
    }
    
    return refined_h_mask, refined_v_mask, phase_outputs


def generate_intelligent_mask(
    refined_horizontal_mask: np.ndarray,
    refined_vertical_mask: np.ndarray,
    cv_image: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Phase 4: Generate intelligent mask that preserves text while removing form lines.
    
    This phase:
    1. Detects text components using connected component analysis
    2. Distinguishes text from lines using geometric properties
    3. Creates exclusion mask for text components
    4. Refines line mask by subtracting text components
    5. Applies intelligent removal to image
    
    Note: Dilation is handled by Phase 5 (Final Mask Generation).
    
    Args:
        refined_horizontal_mask: Refined horizontal mask from Phase 3
        refined_vertical_mask: Refined vertical mask from Phase 3
        cv_image: Original OpenCV image
        config: Preprocessing configuration
        
    Returns:
        Tuple of (initial_mask, cleaned_image, text_exclusion_mask, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Get image dimensions
    if len(cv_image.shape) == 3:
        h, w = cv_image.shape[:2]
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        h, w = cv_image.shape
        gray = cv_image.copy()
    
    image_area = h * w
    
    # Prepare binary image for text detection
    if len(np.unique(gray)) > 2:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.bitwise_not(gray) if np.mean(gray) > 127 else gray
    
    # Step 1: Detect text components using connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Step 2: Classify components as text or line
    text_exclusion_mask = np.zeros_like(binary, dtype=np.uint8)
    text_components = []
    line_components = []
    
    for label_id in range(1, num_labels):  # Skip background (label 0)
        # Stats indices: 0=LEFT, 1=TOP, 2=WIDTH, 3=HEIGHT, 4=AREA
        x = stats[label_id, 0]  # CC_STAT_LEFT
        y = stats[label_id, 1]  # CC_STAT_TOP
        width = stats[label_id, 2]  # CC_STAT_WIDTH
        height = stats[label_id, 3]  # CC_STAT_HEIGHT
        area = stats[label_id, 4]  # CC_STAT_AREA
        
        # Calculate geometric properties
        aspect_ratio = max(width / height, height / width) if min(height, width) > 0 else 0
        bbox_area = width * height
        solidity = area / bbox_area if bbox_area > 0 else 0
        area_ratio = area / image_area
        
        # Classification: is this text?
        is_text = False
        if config.structural_enable_text_preservation:
            is_text = (
                aspect_ratio >= config.structural_text_aspect_ratio_min and
                aspect_ratio <= config.structural_text_aspect_ratio_max and
                area >= config.structural_text_min_area and
                area_ratio <= config.structural_text_max_area_ratio and
                solidity >= config.structural_text_min_solidity
            )
        
        if is_text:
            # Mark as text (exclude from removal)
            text_exclusion_mask[labels == label_id] = 255
            text_components.append({
                "label": int(label_id),
                "area": int(area),
                "bbox": (int(x), int(y), int(width), int(height)),
                "aspect_ratio": float(aspect_ratio),
                "solidity": float(solidity),
            })
        else:
            line_components.append({
                "label": int(label_id),
                "area": int(area),
                "bbox": (int(x), int(y), int(width), int(height)),
            })
    
    # Store text components visualization
    text_vis = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    text_vis[text_exclusion_mask > 0] = [0, 255, 0]  # Green for text
    text_vis_pil = Image.fromarray(cv2.cvtColor(text_vis, cv2.COLOR_BGR2RGB))
    phase_outputs["text_components"] = {
        "image": image_to_base64(text_vis_pil),
        "size": text_vis_pil.size,
        "count": len(text_components),
    }
    
    # Step 3: Combine refined masks
    combined_line_mask = cv2.bitwise_or(refined_horizontal_mask, refined_vertical_mask)
    
    # Step 4: Refine line mask by excluding text components
    # Invert text exclusion mask (text = 0, background = 255)
    text_inverse = cv2.bitwise_not(text_exclusion_mask)
    
    # Only keep line pixels that are NOT part of text
    # Note: Dilation will be applied in Phase 5 (Final Mask Generation)
    initial_mask = cv2.bitwise_and(combined_line_mask, text_inverse)
    
    # Store initial mask visualization (before dilation)
    initial_mask_pil = Image.fromarray(initial_mask)
    phase_outputs["initial_mask"] = {
        "image": image_to_base64(initial_mask_pil),
        "size": initial_mask_pil.size,
    }
    
    # Step 5: Apply intelligent removal (using initial mask for preview)
    # Note: Final cleaning will use the dilated mask from Phase 5
    # Determine background color
    unique_values, counts = np.unique(gray, return_counts=True)
    background_color = unique_values[np.argmax(counts)]
    
    # Create cleaned image (preview with initial mask)
    cleaned_gray = gray.copy()
    cleaned_gray[initial_mask > 0] = background_color
    
    # Convert back to original format
    if len(cv_image.shape) == 3:
        cleaned_image = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)
    else:
        cleaned_image = cleaned_gray
    
    # Store before/after comparison
    before_after = np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cleaned_image])
    before_after_pil = Image.fromarray(cv2.cvtColor(before_after, cv2.COLOR_BGR2RGB))
    phase_outputs["before_after"] = {
        "image": image_to_base64(before_after_pil),
        "size": before_after_pil.size,
    }
    
    # Statistics
    line_pixels = np.sum(initial_mask > 0)
    text_pixels = np.sum(text_exclusion_mask > 0)
    
    phase_outputs["statistics"] = {
        "text_components_count": len(text_components),
        "line_components_count": len(line_components),
        "text_pixels": int(text_pixels),
        "line_pixels_detected": int(line_pixels),
        "text_preservation_enabled": config.structural_enable_text_preservation,
    }
    
    return initial_mask, cleaned_image, text_exclusion_mask, phase_outputs


def _create_comb_field_mask(
    refined_h_mask: np.ndarray,
    refined_v_mask: np.ndarray,
    comb_fields: list[dict],
    h: int,
    w: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Create mask containing ONLY lines from comb fields.
    
    Enhanced extraction with expanded bounding boxes and morphological operations
    to ensure complete removal of comb field structures. This function is used for
    selective comb field removal while preserving other form structures (boxes, standalone lines).
    
    Args:
        refined_h_mask: Refined horizontal lines mask
        refined_v_mask: Refined vertical lines mask
        comb_fields: List of detected comb field dictionaries with keys:
            top_y, bottom_y, left_x, right_x, x_coordinates
        h: Image height
        w: Image width
        config: Preprocessing configuration
        
    Returns:
        Binary mask containing only lines from comb fields
    """
    comb_field_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get padding from config
    padding = config.structural_comb_field_extraction_padding
    tolerance = config.structural_comb_field_vertical_tolerance
    
    for comb_field in comb_fields:
        top_y = comb_field.get("top_y", 0)
        bottom_y = comb_field.get("bottom_y", h)
        left_x = comb_field.get("left_x", 0)
        right_x = comb_field.get("right_x", w)
        
        # Expand bounding box with padding to ensure we catch all edge lines
        expanded_top_y = max(0, top_y - padding - tolerance)
        expanded_bottom_y = min(h, bottom_y + padding + tolerance + 1)
        expanded_left_x = max(0, left_x - padding)
        expanded_right_x = min(w, right_x + padding + 1)
        
        # Ensure coordinates are within bounds
        expanded_top_y = max(0, min(expanded_top_y, h - 1))
        expanded_bottom_y = max(0, min(expanded_bottom_y, h - 1))
        expanded_left_x = max(0, min(expanded_left_x, w - 1))
        expanded_right_x = max(0, min(expanded_right_x, w - 1))
        
        # Extract ALL horizontal lines in the expanded comb field region
        # Range: [expanded_top_y : expanded_bottom_y] within [expanded_left_x : expanded_right_x]
        if expanded_bottom_y > expanded_top_y and expanded_right_x > expanded_left_x:
            horizontal_region = refined_h_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x]
            if np.any(horizontal_region > 0):
                comb_field_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x] = np.maximum(
                    comb_field_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x],
                    horizontal_region
                )
        
        # Extract ALL vertical lines in the expanded comb field region
        # Range: [expanded_left_x : expanded_right_x] within [expanded_top_y : expanded_bottom_y]
        if expanded_bottom_y > expanded_top_y and expanded_right_x > expanded_left_x:
            vertical_region = refined_v_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x]
            if np.any(vertical_region > 0):
                comb_field_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x] = np.maximum(
                    comb_field_mask[expanded_top_y:expanded_bottom_y, expanded_left_x:expanded_right_x],
                    vertical_region
                )
    
    # Apply morphological operations to ensure complete line capture
    # Small dilation to connect any broken line segments
    if len(comb_fields) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        comb_field_mask = cv2.dilate(comb_field_mask, kernel, iterations=1)
        comb_field_mask = cv2.erode(comb_field_mask, kernel, iterations=1)
    
    return comb_field_mask


def generate_final_clean_zone_mask(
    refined_horizontal_mask: np.ndarray,
    refined_vertical_mask: np.ndarray,
    text_exclusion_mask: np.ndarray,
    cv_image: np.ndarray,
    config: PreprocessingConfig,
    detected_boxes: list[dict] | None = None,
    comb_fields: list[dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Phase 5: Generate final unified binary mask (The Clean Zone Map).
    
    This phase creates the final "Search and Destroy" map that represents
    everything that is NOT data. The mask is dilated to ensure complete
    removal of form lines without leaving "ghost lines" behind.
    
    This phase:
    1. Combines all refined horizontal and vertical masks
    2. Excludes text components using text exclusion mask from Phase 4
    3. Applies configurable dilation to thicken detected lines (1-2 pixels)
    4. Creates visualization showing the "Clean Zone Map"
    5. Calculates statistics (mask coverage, pixel counts, etc.)
    
    Args:
        refined_horizontal_mask: Refined horizontal mask from Phase 3
        refined_vertical_mask: Refined vertical mask from Phase 3
        text_exclusion_mask: Text exclusion mask from Phase 4
        cv_image: Original OpenCV image for dimension reference
        config: Preprocessing configuration
        detected_boxes: List of detected boxes from Phase 3 (for visualization/statistics)
        comb_fields: List of detected comb fields from Phase 3 (for visualization/statistics)
        
    Returns:
        Tuple of (final_clean_zone_mask, phase_outputs_dict)
    """
    phase_outputs = {}
    
    # Get image dimensions
    if len(cv_image.shape) == 3:
        h, w = cv_image.shape[:2]
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        h, w = cv_image.shape
        gray = cv_image.copy()
    
    image_area = h * w
    
    # Step 1: Create mask based on removal mode
    if config.structural_comb_field_removal_only and comb_fields:
        # Comb-field-only removal: Only remove lines from detected comb fields
        combined_line_mask = _create_comb_field_mask(
            refined_horizontal_mask,
            refined_vertical_mask,
            comb_fields,
            h,
            w,
            config,
        )
        comb_fields_processed = len(comb_fields) if comb_fields else 0
    else:
        # Full removal: Combine all refined horizontal and vertical masks (default behavior)
        combined_line_mask = cv2.bitwise_or(refined_horizontal_mask, refined_vertical_mask)
        comb_fields_processed = 0
    
    # Step 2: Exclude text components using text exclusion mask from Phase 4
    # Invert text exclusion mask (text = 0, background = 255)
    text_inverse = cv2.bitwise_not(text_exclusion_mask)
    
    # Only keep line pixels that are NOT part of text
    final_mask = cv2.bitwise_and(combined_line_mask, text_inverse)
    
    # Step 3: Apply configurable dilation to thicken detected lines
    # Ensure kernel size is odd
    kernel_size = config.structural_mask_dilation_kernel_size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create dilation kernel
    dilation_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Apply dilation with configured iterations
    dilated_mask = cv2.dilate(
        final_mask,
        dilation_kernel,
        iterations=config.structural_mask_dilation_iterations
    )
    
    # The dilated mask is our final "Clean Zone Map"
    clean_zone_mask = dilated_mask
    
    # Step 4: Create visualizations
    # Visualization 1: Clean Zone Map (final mask)
    clean_zone_pil = Image.fromarray(clean_zone_mask)
    phase_outputs["clean_zone_mask"] = {
        "image": image_to_base64(clean_zone_pil),
        "size": clean_zone_pil.size,
        "description": "Final unified binary mask representing everything that is NOT data (Clean Zone Map)"
    }
    
    # Visualization 2: Overlay on original image
    overlay_vis = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    # Create colored overlay (red for areas to be removed)
    overlay_vis[clean_zone_mask > 0] = [0, 0, 255]  # Red overlay
    # Blend with original (semi-transparent)
    overlay_vis = cv2.addWeighted(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        0.7,
        overlay_vis,
        0.3,
        0
    )
    overlay_vis_pil = Image.fromarray(cv2.cvtColor(overlay_vis, cv2.COLOR_BGR2RGB))
    phase_outputs["clean_zone_overlay"] = {
        "image": image_to_base64(overlay_vis_pil),
        "size": overlay_vis_pil.size,
        "description": "Clean Zone Map overlayed on original image (red = areas to be removed)"
    }
    
    # Visualization 3: Before/After dilation comparison
    before_after_dilation = np.hstack([final_mask, dilated_mask])
    before_after_dilation_pil = Image.fromarray(before_after_dilation)
    phase_outputs["dilation_comparison"] = {
        "image": image_to_base64(before_after_dilation_pil),
        "size": before_after_dilation_pil.size,
        "description": "Before (left) and after (right) dilation"
    }
    
    # Step 5: Calculate statistics
    mask_pixels = np.sum(clean_zone_mask > 0)
    mask_coverage_percentage = (mask_pixels / image_area) * 100
    
    # Calculate dilation effect
    before_dilation_pixels = np.sum(final_mask > 0)
    after_dilation_pixels = np.sum(dilated_mask > 0)
    dilation_increase_pixels = after_dilation_pixels - before_dilation_pixels
    dilation_increase_percentage = (
        (dilation_increase_pixels / before_dilation_pixels * 100)
        if before_dilation_pixels > 0 else 0.0
    )
    
    phase_outputs["statistics"] = {
        "removal_mode": "comb_fields_only" if config.structural_comb_field_removal_only else "full",
        "mask_pixels": int(mask_pixels),
        "mask_coverage_percentage": float(mask_coverage_percentage),
        "before_dilation_pixels": int(before_dilation_pixels),
        "after_dilation_pixels": int(after_dilation_pixels),
        "dilation_increase_pixels": int(dilation_increase_pixels),
        "dilation_increase_percentage": float(dilation_increase_percentage),
        "dilation_kernel_size": int(kernel_size),
        "dilation_iterations": int(config.structural_mask_dilation_iterations),
        "target_dilation_pixels": int(config.structural_mask_dilation_pixels),
    }
    
    # Add comb field removal statistics if applicable
    if config.structural_comb_field_removal_only:
        phase_outputs["statistics"]["comb_fields_processed"] = comb_fields_processed
    
    phase_outputs["description"] = (
        "Final unified binary mask (Clean Zone Map) representing everything that is NOT data. "
        "This mask is used as the 'Search and Destroy' map for removing form lines while preserving text."
    )
    
    return clean_zone_mask, phase_outputs

