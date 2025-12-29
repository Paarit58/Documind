"""
Image encoding utilities.

Converts PIL Images to base64 data URLs for web display.
"""

import base64
import io

from PIL import Image


def image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URL.
    
    Args:
        pil_image: PIL Image
        
    Returns:
        Base64 data URL string
    """
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"



