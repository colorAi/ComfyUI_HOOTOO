# ComfyUI/custom_nodes/comfyui-HOOTOO/__init__.py

from .image_3d_transform import NODE_CLASS_MAPPINGS as ImageTransform_NODE_CLASS_MAPPINGS
from .ImageTransformSegment import NODE_CLASS_MAPPINGS as ImageTransformSegment_NODE_CLASS_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **ImageTransform_NODE_CLASS_MAPPINGS,
    **ImageTransformSegment_NODE_CLASS_MAPPINGS
}

WEB_DIRECTORY = "./web" # 在这里只定义一次 WEB_DIRECTORY

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]

print("HOOTOO ComfyUI Plugin loaded. (image_3d_transform node available)")
print("HOOTOO ComfyUI Plugin loaded. (ImageTransformSegment node available)")