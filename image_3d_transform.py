import torch
import numpy as np
from PIL import Image, ImageDraw, ImageTransform
import math
import cv2
import os
import tempfile

class HOOTOO_ImageTransform:
    """兼容2D/3D的稳定版图像变换 - 输出尺寸始终固定为原始图像尺寸"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("INT", {"default": 10, "min": 1, "max": 9999}),
                # 输出尺寸参数
                "output_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                # 共用参数
                "translate_x": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "translate_y": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                # 新增的动画缩放参数
                "scale_animate_start": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "scale_animate_end": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                # 3D专用
                "rotate_x": ("FLOAT", {"default": 0.0, "min": -179.0, "max": 179.0, "step": 0.5}), 
                "rotate_y": ("FLOAT", {"default": 0.0, "min": -179.0, "max": 179.0, "step": 0.5}), 
                "rotate_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5}),
                "focal_length": ("FLOAT", {"default": 1000.0, "min": 10.0, "max": 5000.0, "step": 10.0}),
                "bg_color": (["black", "white", "checker"], {"default": "black"}), # 这个bg_color将作为备用
                "output_padding": ("INT", {"default": 0, "min": 0, "max": 500}), 
            },
            "optional": { 
                "image": ("IMAGE",),
                "background_image": ("IMAGE",), # 新增可选背景图像输入
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "transform"
    CATEGORY = "HOOTOO/image"

    DEFAULT_CHECKERBOARD_OPERATION_SIZE = 2048 

    def transform(self, frames, output_width, output_height, translate_x, translate_y, 
                  scale_animate_start, scale_animate_end,
                  rotate_x, rotate_y, rotate_z, focal_length, bg_color, output_padding,
                  image=None, background_image=None): # 接收 background_image

        # --- 确定 img_orig (要变换的前景图像) ---
        if image is not None and image.shape[0] > 0:
            img_orig = self.tensor2pil(image[0])
            # 如果有输入图像，最终输出尺寸固定为输入图像的尺寸
            fixed_output_width, fixed_output_height = img_orig.size
            print(f"Image input detected. Output size will be {fixed_output_width}x{fixed_output_height} (from input image).")
        else:
            # 没有图像输入，生成一个大棋盘格作为内部操作对象
            print(f"No image input detected. Using internal checkerboard of size {self.DEFAULT_CHECKERBOARD_OPERATION_SIZE}x{self.DEFAULT_CHECKERBOARD_OPERATION_SIZE}.")
            img_orig = self._create_checkerboard_alpha_background(self.DEFAULT_CHECKERBOARD_OPERATION_SIZE, self.DEFAULT_CHECKERBOARD_OPERATION_SIZE)
            # 如果没有输入图像，最终输出尺寸使用用户设置的 output_width/height
            fixed_output_width, fixed_output_height = output_width, output_height
            print(f"Output size will be {fixed_output_width}x{fixed_output_height} (from node parameters).")
            # 如果是默认棋盘格操作，通常期望最终背景也是棋盘格，以便无限延伸效果
            # 注意：此处 bg_color 只是备用，如果 background_image 有连接，它将被覆盖
            if bg_color != "checker" and (background_image is None or background_image.shape[0] == 0):
                print("Warning: No image or background image input. For seamless checkerboard, 'bg_color' should ideally be 'checker'.")

        image_sequence = []
        mask_sequence = []

        # --- 准备背景图像序列 (如果提供了 background_image) ---
        bg_img_sequence_pil = []
        if background_image is not None and background_image.shape[0] > 0:
            # 检查 background_image 的帧数是否与 frames 匹配
            if background_image.shape[0] != frames:
                print(f"Warning: Background image has {background_image.shape[0]} frames, but 'frames' parameter is {frames}. Using only the first frame of background image or tiling if less.")
                # 这里可以选择更复杂的处理，如循环、拉伸，但简单起见，我们循环使用背景帧
                for i in range(frames):
                    bg_img_sequence_pil.append(self.tensor2pil(background_image[i % background_image.shape[0]]))
            else:
                for i in range(frames):
                    bg_img_sequence_pil.append(self.tensor2pil(background_image[i]))
            
            # 检查背景图像尺寸是否与固定输出尺寸匹配，如果不匹配，可能会有拉伸
            if bg_img_sequence_pil[0].size != (fixed_output_width, fixed_output_height):
                print(f"Warning: Background image size {bg_img_sequence_pil[0].size} does not match target output size {fixed_output_width}x{fixed_output_height}. Background will be resized.")


        for frame_idx in range(frames):
            progress = frame_idx / max(frames-1, 1)

            current_animated_scale = scale_animate_start + (scale_animate_end - scale_animate_start) * progress
            final_scale = current_animated_scale 

            current_translate_x = translate_x * progress
            current_translate_y = translate_y * progress
            current_rotate_x = rotate_x * progress
            current_rotate_y = rotate_y * progress
            current_rotate_z = rotate_z * progress
            
            # 传递当前的背景图像（如果存在）
            current_frame_bg_pil = None
            if bg_img_sequence_pil:
                current_frame_bg_pil = bg_img_sequence_pil[frame_idx]
                # 确保背景图像是 RGBA 且尺寸匹配
                if current_frame_bg_pil.mode != 'RGBA':
                    current_frame_bg_pil = current_frame_bg_pil.convert('RGBA')
                if current_frame_bg_pil.size != (fixed_output_width, fixed_output_height):
                    current_frame_bg_pil = current_frame_bg_pil.resize((fixed_output_width, fixed_output_height), Image.LANCZOS)


            result_img_pil, result_mask_pil = self._transform_3d( 
                img_orig.copy(), 
                translation=(current_translate_x, current_translate_y),
                scale=final_scale, 
                rotation_3d=(current_rotate_x, current_rotate_y, current_rotate_z),
                focal_length=focal_length,
                bg_color=bg_color, # 这个bg_color现在是当 background_image 不存在时的备用
                output_padding=output_padding, 
                fixed_output_size=(fixed_output_width, fixed_output_height),
                external_background_pil=current_frame_bg_pil # 传递外部背景
            )

            # Convert result_img_pil to RGB before converting to tensor
            if result_img_pil.mode == 'RGBA':
                result_img_pil = result_img_pil.convert('RGB')

            image_sequence.append(self.pil2tensor(result_img_pil))
            mask_sequence.append(self.pil2mask_tensor(result_mask_pil)) 

        return (torch.cat(image_sequence, dim=0), torch.cat(mask_sequence, dim=0))

    def _create_checkerboard_alpha_background(self, width, height, size=32):
        """创建用于默认图像的棋盘格，背景透明"""
        bg = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(bg)
        # 深色格子和浅色格子
        color_dark = (50, 50, 50, 255) # 完全不透明
        color_light = (200, 200, 200, 255) # 完全不透明

        for y in range(0, height, size):
            for x in range(0, width, size):
                if (x//size + y//size) % 2 == 0:
                    color = color_light 
                else:
                    color = color_dark 
                draw.rectangle([x,y,x+size,y+size], fill=color)
        return bg


    def _transform_3d(self, img, translation, scale, rotation_3d, focal_length, bg_color, output_padding, fixed_output_size, external_background_pil=None):
        """安全的3D变换实现 (输出尺寸始终固定)"""
        
        w_orig_input_img, h_orig_input_img = fixed_output_size 

        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        original_alpha_pil = img.getchannel('A').copy() 

        # 1. 应用初始缩放
        if scale != 1.0:
            new_w = max(1, int(img.size[0] * scale))
            new_h = max(1, int(img.size[1] * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            original_alpha_pil = original_alpha_pil.resize((new_w, new_h), Image.LANCZOS) 

        rot_x, rot_y, rot_z = rotation_3d

        # 2. Z轴旋转
        if rot_z != 0:
            fill_color = self._get_fill_color_tuple(bg_color, is_pil_fill=True)
            img = img.rotate(rot_z, expand=True, resample=Image.BICUBIC, fillcolor=fill_color)
            
            original_alpha_pil = original_alpha_pil.rotate(rot_z, expand=True, resample=Image.BICUBIC, fillcolor=0) 

        # 3. 3D透视 (X 和 Y 轴)
        if any(r != 0 for r in (rot_x, rot_y)):
            img_transformed_for_color = self._apply_perspective_opencv(img, (rot_x, rot_y), focal_length, bg_color)
            if img_transformed_for_color.mode != 'RGBA':
                img_transformed_for_color = img_transformed_for_color.convert('RGBA')
            
            temp_alpha_rgba_for_transform = Image.new('RGBA', original_alpha_pil.size, (0, 0, 0, 0)) 
            temp_alpha_rgba_for_transform.putalpha(original_alpha_pil) 
            
            alpha_transformed_pil_rgba = self._apply_perspective_opencv(temp_alpha_rgba_for_transform, (rot_x, rot_y), focal_length, "black") 
            alpha_transformed_pil = alpha_transformed_pil_rgba.getchannel('A') 

            img = img_transformed_for_color 
            original_alpha_pil = alpha_transformed_pil 

        # --- 创建固定大小的背景画布 (根据 external_background_pil 或 bg_color) ---
        if external_background_pil:
            bg = external_background_pil # 使用传入的外部背景
        else:
            bg = self._create_bg(w_orig_input_img, h_orig_input_img, bg_color)
        
        # 对于 Mask 的背景 (mask_bg) 必须是全透明的
        mask_bg = Image.new('L', (w_orig_input_img, h_orig_input_img), 0) 

        current_img_w, current_img_h = img.size 
        
        effective_w = w_orig_input_img - 2 * output_padding
        effective_h = h_orig_input_img - 2 * output_padding

        if effective_w <= 0 or effective_h <= 0:
            print(f"Warning: output_padding ({output_padding}) is too large for image size ({w_orig_input_img}, {h_orig_input_img}). Image will be empty or severely cropped.")
            return Image.new("RGBA", (w_orig_input_img, h_orig_input_img), self._get_fill_color_tuple(bg_color, is_pil_fill=True)), \
                   Image.new("L", (w_orig_input_img, h_orig_input_img), 0)

        pos_in_effective_x = (effective_w - current_img_w) // 2
        pos_in_effective_y = (effective_h - current_img_h) // 2

        pos_x = output_padding + pos_in_effective_x
        pos_y = output_padding + pos_in_effective_y
        
        pos_x += int(translation[0])
        pos_y += int(translation[1])
        
        # 将前景图像粘贴到背景上
        bg.paste(img, (pos_x, pos_y), img)
        mask_bg.paste(original_alpha_pil, (pos_x, pos_y), original_alpha_pil)

        return bg, mask_bg

    def _create_bg(self, width, height, bg_color):
        """创建背景画布，支持 RGBA"""
        if bg_color == "checker":
            return self._create_checkerboard_visible(width, height)
        fill_color = self._get_fill_color_tuple(bg_color, is_pil_fill=True)
        return Image.new("RGBA", (width, height), fill_color)

    def _create_checkerboard_visible(self, width, height, size=32):
        """创建用于可见背景的棋盘格，格子为不透明"""
        bg = Image.new("RGBA", (width, height)) 
        draw = ImageDraw.Draw(bg)
        for y in range(0, height, size):
            for x in range(0, width, size):
                if (x//size + y//size) % 2 == 0:
                    color = (200, 200, 200, 255) 
                else:
                    color = (50, 50, 50, 255) 
                draw.rectangle([x,y,x+size,y+size], fill=color)
        return bg

    def _get_fill_color_tuple(self, bg_color, is_pil_fill=False):
        """辅助函数：根据背景颜色字符串获取 RGB 或 RGBA 元组
           is_pil_fill: 如果为 True，返回 PIL 兼容的 RGBA (或 RGB) 元组
                        如果为 False，返回 OpenCV 兼容的 BGRA (或 BGR) 元组
        """
        if is_pil_fill:
            if bg_color == "white":
                return (255, 255, 255, 255) 
            elif bg_color == "black":
                return (0, 0, 0, 255)     
            elif bg_color == "checker":
                return (0, 0, 0, 0) 
            return (0, 0, 0, 255) 
        else: # OpenCV BGRA
            if bg_color == "white":
                return (255, 255, 255, 255) 
            elif bg_color == "black":
                return (0, 0, 0, 255)
            elif bg_color == "checker":
                return (0, 0, 0, 0) 
            return (0, 0, 0, 255)

    def _apply_perspective_opencv(self, img, rotation, focal_length, bg_color):
        """
        使用OpenCV应用更真实的透视变形
        计算方法基于3D旋转和透视投影原理，并动态调整输出画布尺寸
        此函数返回的图像是完全包含透视变形结果的，大小可能大于原始输入img
        """
        w, h = img.size
        rot_x, rot_y = rotation

        img_np = np.array(img)
        if img_np.shape[-1] == 4: 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
        elif img_np.shape[-1] == 3: 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        rad_x = math.radians(rot_x)
        rad_y = math.radians(rot_y)

        pts3d_original = np.array([
            [-w/2, -h/2, 0], 
            [ w/2, -h/2, 0], 
            [ w/2,  h/2, 0], 
            [-w/2,  h/2, 0]  
        ], dtype=np.float32)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rad_x), -math.sin(rad_x)],
            [0, math.sin(rad_x), math.cos(rad_x)]
        ], dtype=np.float32)

        Ry = np.array([
            [math.cos(rad_y), 0, math.sin(rad_y)],
            [0, 1, 0],
            [-math.sin(rad_y), 0, math.cos(rad_y)]
        ], dtype=np.float32)

        R = np.dot(Ry, Rx) 

        pts3d_rotated = np.dot(pts3d_original, R.T)

        pts2_projected = np.zeros((4, 2), dtype=np.float32)
        for i, p_rot in enumerate(pts3d_rotated):
            x_rot, y_rot, z_rot = p_rot

            denominator = (focal_length + z_rot)
            if denominator <= 0:
                print(f"Warning: Image point {i} went behind the camera. Perspective may be invalid. z_rot: {z_rot}, focal_length: {focal_length}")
                denominator = 1e-6 

            x_projected = (focal_length * x_rot) / denominator
            y_projected = (focal_length * y_rot) / denominator

            pts2_projected[i, 0] = x_projected
            pts2_projected[i, 1] = y_projected

        min_x = np.min(pts2_projected[:, 0])
        max_x = np.max(pts2_projected[:, 0])
        min_y = np.min(pts2_projected[:, 1])
        max_y = np.max(pts2_projected[:, 1])

        output_w = int(round(max_x - min_x))
        output_h = int(round(max_y - min_y))

        output_w = max(output_w, 1)
        output_h = max(output_h, 1)

        translation_offset_x = -min_x
        translation_offset_y = -min_y

        pts2_final = pts2_projected.copy()
        pts2_final[:, 0] += translation_offset_x
        pts2_final[:, 1] += translation_offset_y

        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        opencv_fill_color = self._get_fill_color_tuple(bg_color, is_pil_fill=False)
        
        try:
            matrix = cv2.getPerspectiveTransform(pts1, pts2_final)
            transformed_img_np = cv2.warpPerspective(img_np, matrix, (output_w, output_h),
                                                      borderMode=cv2.BORDER_CONSTANT,
                                                      borderValue=opencv_fill_color,
                                                      flags=cv2.INTER_CUBIC)
        except cv2.error as e:
            print(f"OpenCV Error in _apply_perspective_opencv: {e}")
            print(f"Input image size: ({w}, {h})")
            print(f"Projected 2D points (before offset):\n{pts2_projected}")
            print(f"Final 2D points (pts2_final):\n{pts2_final}")
            print(f"Calculated output size: ({output_w}, {output_h})")
            return Image.new("RGBA", (w, h), self._get_fill_color_tuple(bg_color, is_pil_fill=True))

        if transformed_img_np.shape[-1] == 4: 
            return Image.fromarray(cv2.cvtColor(transformed_img_np, cv2.COLOR_BGRA2RGBA))
        else: 
            return Image.fromarray(cv2.cvtColor(transformed_img_np, cv2.COLOR_BGR2RGB)).convert('RGBA')


    @staticmethod
    def tensor2pil(image):
        """Tensor转PIL图像，支持 RGBA"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.shape[-1] == 4:
            return Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8), 'RGBA')
        else:
            rgb_img = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8), 'RGB')
            return rgb_img.convert('RGBA')


    @staticmethod
    def pil2tensor(image):
        """PIL图像转Tensor，确保输出为RGB"""
        if image.mode == 'RGBA':
            image = image.convert('RGB') # Convert to RGB if it's RGBA
        
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def pil2mask_tensor(mask_image):
        """将 PIL 单通道图像（如Alpha通道）转换为 ComfyUI MASK Tensor"""
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "HOOTOO_ImageTransform": HOOTOO_ImageTransform
}

WEB_DIRECTORY = "./web"