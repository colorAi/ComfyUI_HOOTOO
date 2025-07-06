import torch
import numpy as np
from PIL import Image, ImageDraw, ImageTransform
import math
import cv2
import os
import tempfile

# 定义一个辅助类型，用于在 ComfyUI 内部表示我们的复杂状态
TRANSFORM_STATE_TYPE = ("TRANSFORM_STATE",) 

class HOOTOO_ImageTransformSegment:
    """支持分段动画的图像变换节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), # 图像输入，用于链式连接
                # 将 initial_transform_state 放在这里
                "initial_transform_state": (TRANSFORM_STATE_TYPE, {"default": None}), # 默认为None，表示未连接或第一个节点
                "total_frames": ("INT", {"default": 500, "min": 1, "max": 9999}), # 总帧数，通常由第一个节点设置
                "segment_start_frame": ("INT", {"default": 0, "min": 0, "max": 9998}), # 本段开始帧 (包含)
                "segment_end_frame": ("INT", {"default": 100, "min": 1, "max": 9999}), # 本段结束帧 (包含)

                # 确保 scale_start 被添加回来
                "scale_start": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}), # 新增回来的参数

                # 独立的，用于本段动画结束时的参数设置 (依然是用户手动设置)
                "translate_x_end": ("INT", {"default": 100, "min": -5000, "max": 5000}),
                "translate_y_end": ("INT", {"default": 0, "min": -5000, "max": 5000}),
                "scale_end": ("FLOAT", {"default": 1.2, "min": 0.01, "max": 10.0, "step": 0.01}),
                "rotate_x_end": ("FLOAT", {"default": 0.0, "min": -179.0, "max": 179.0, "step": 0.5}), 
                "rotate_y_end": ("FLOAT", {"default": 0.0, "min": -179.0, "max": 179.0, "step": 0.5}), 
                "rotate_z_end": ("FLOAT", {"default": 30.0, "min": -360.0, "max": 360.0, "step": 0.5}),
                
                "focal_length": ("FLOAT", {"default": 1000.0, "min": 10.0, "max": 5000.0, "step": 10.0}),
                "bg_color": (["black", "white", "checker"], {"default": "black"}),
                "output_padding": ("INT", {"default": 0, "min": 0, "max": 500}), 
            },
            "optional": { 
                "background_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", 
                    TRANSFORM_STATE_TYPE) # 新增的复合输出接口，用于传递本段的结束状态给下一个节点

    RETURN_NAMES = ("IMAGE_SEQUENCE_SEGMENT", "MASK_SEQUENCE_SEGMENT",
                    "FINAL_TRANSFORM_STATE")

    FUNCTION = "transform_segment"
    CATEGORY = "HOOTOO/image"

    DEFAULT_CHECKERBOARD_OPERATION_SIZE = 2048 

    def transform_segment(self, image, total_frames, segment_start_frame, segment_end_frame,
                          initial_transform_state, # 接收复合输入
                          scale_start, # 确保在函数签名中也添加回来
                          translate_x_end, translate_y_end, scale_end,
                          rotate_x_end, rotate_y_end, rotate_z_end,
                          focal_length, bg_color, output_padding,
                          background_image=None):

        # 定义默认的起始状态，用于第一个节点或未连接initial_transform_state的情况
        # 注意：这里我们不再需要 default_scale_start，因为它现在作为独立的参数存在
        default_translate_x_start = 0
        default_translate_y_start = 0
        default_rotate_x_start = 0.0
        default_rotate_y_start = 0.0
        default_rotate_z_start = 0.0
        default_output_width = 512 
        default_output_height = 512 

        # 根据 initial_transform_state 解包起始参数
        if initial_transform_state is not None and len(initial_transform_state) == 8: # 确保元组长度正确
            (translate_x_upstream, translate_y_upstream, scale_upstream, # 从上游获取的scale_start
             rotate_x_upstream, rotate_y_upstream, rotate_z_upstream,
             input_width_from_upstream, input_height_from_upstream) = initial_transform_state
        else:
            # 如果没有连接或连接了错误类型，则使用默认起始值
            print("No upstream transform state connected. Using default start parameters.")
            translate_x_upstream = default_translate_x_start
            translate_y_upstream = default_translate_y_start
            scale_upstream = 1.0 # 默认的 scale_start
            rotate_x_upstream = default_rotate_x_start
            rotate_y_upstream = default_rotate_y_start
            rotate_z_upstream = default_rotate_z_start
            input_width_from_upstream = None 
            input_height_from_upstream = None

        # 确定本段使用的起始参数
        # 如果 initial_transform_state 被连接，则使用上游的值作为起始
        # 否则，使用节点自身的 scale_start 参数
        used_translate_x_start = translate_x_upstream
        used_translate_y_start = translate_y_upstream
        used_scale_start = scale_upstream
        used_rotate_x_start = rotate_x_upstream
        used_rotate_y_start = rotate_y_upstream
        used_rotate_z_start = rotate_z_upstream

        # 特别处理第一个节点的情况：如果 initial_transform_state 为 None (即没有连接上游)
        # 则使用节点自己面板上设定的 scale_start 值。
        # 对于其他参数（translate, rotate），目前的设计是如果 initial_transform_state 为 None
        # 则它们会使用上面定义的 default_..._start 值。
        # 如果你希望所有起始参数在未连接时都使用节点上的面板值，需要更复杂的逻辑。
        # 但考虑到我们期望通过 initial_transform_state 传递全部状态，
        # 这种设计意味着 initial_transform_state 应该始终被连接（至少是第一个初始状态节点）。
        if initial_transform_state is None:
             used_scale_start = scale_start # 如果没有上游状态，就使用节点自己的 scale_start

        # 确定最终输出尺寸的逻辑 (优先顺序：输入图像尺寸 > 上游状态尺寸 > 默认值)
        fixed_output_width = default_output_width
        fixed_output_height = default_output_height

        if image is not None and image.shape[0] > 0:
            img_orig = self.tensor2pil(image[0])
            fixed_output_width, fixed_output_height = img_orig.size
            print(f"Image input detected. Output size will be {fixed_output_width}x{fixed_output_height} (from input image).")
        elif input_width_from_upstream is not None and input_height_from_upstream is not None:
            fixed_output_width = input_width_from_upstream
            fixed_output_height = input_height_from_upstream
            print(f"No direct image input. Output size will be {fixed_output_width}x{fixed_output_height} (from upstream state).")
        else:
            print(f"No image input and no upstream size. Output size will be {fixed_output_width}x{fixed_output_height} (from node defaults).")
            if bg_color != "checker" and (background_image is None or background_image.shape[0] == 0):
                print("Warning: No image or background image input. For seamless checkerboard, 'bg_color' should ideally be 'checker'.")
            img_orig = self._create_checkerboard_alpha_background(self.DEFAULT_CHECKERBOARD_OPERATION_SIZE, self.DEFAULT_CHECKERBOARD_OPERATION_SIZE)


        # 确保本段帧范围在总帧数内且有效
        segment_start_frame = max(0, min(segment_start_frame, total_frames - 1))
        segment_end_frame = max(segment_start_frame + 1, min(segment_end_frame, total_frames))
        
        image_sequence = []
        mask_sequence = []

        bg_img_sequence_pil = []
        if background_image is not None and background_image.shape[0] > 0:
            for i in range(total_frames): 
                bg_img_sequence_pil.append(self.tensor2pil(background_image[i % background_image.shape[0]]))
            
            if bg_img_sequence_pil[0].size != (fixed_output_width, fixed_output_height):
                print(f"Warning: Background image size {bg_img_sequence_pil[0].size} does not match target output size {fixed_output_width}x{fixed_output_height}. Background will be resized.")

        segment_duration = max(1, segment_end_frame - segment_start_frame)

        for frame_idx_global in range(segment_start_frame, segment_end_frame):
            progress_in_segment = (frame_idx_global - segment_start_frame) / segment_duration

            # 使用确定好的起始参数进行插值
            current_translate_x = used_translate_x_start + (translate_x_end - used_translate_x_start) * progress_in_segment
            current_translate_y = used_translate_y_start + (translate_y_end - used_translate_y_start) * progress_in_segment
            current_scale = used_scale_start + (scale_end - used_scale_start) * progress_in_segment
            current_rotate_x = used_rotate_x_start + (rotate_x_end - used_rotate_x_start) * progress_in_segment
            current_rotate_y = used_rotate_y_start + (rotate_y_end - used_rotate_y_start) * progress_in_segment
            current_rotate_z = used_rotate_z_start + (rotate_z_end - used_rotate_z_start) * progress_in_segment
            
            current_frame_bg_pil = None
            if bg_img_sequence_pil:
                current_frame_bg_pil = bg_img_sequence_pil[frame_idx_global] 
                if current_frame_bg_pil.mode != 'RGBA':
                    current_frame_bg_pil = current_frame_bg_pil.convert('RGBA')
                if current_frame_bg_pil.size != (fixed_output_width, fixed_output_height):
                    current_frame_bg_pil = current_frame_bg_pil.resize((fixed_output_width, fixed_output_height), Image.LANCZOS)

            result_img_pil, result_mask_pil = self._transform_3d( 
                img_orig.copy(), # 始终基于原始图像进行变换
                translation=(current_translate_x, current_translate_y),
                scale=current_scale, 
                rotation_3d=(current_rotate_x, current_rotate_y, current_rotate_z),
                focal_length=focal_length,
                bg_color=bg_color,
                output_padding=output_padding, 
                fixed_output_size=(fixed_output_width, fixed_output_height),
                external_background_pil=current_frame_bg_pil
            )

            if result_img_pil.mode == 'RGBA':
                result_img_pil = result_img_pil.convert('RGB')

            image_sequence.append(self.pil2tensor(result_img_pil))
            mask_sequence.append(self.pil2mask_tensor(result_mask_pil)) 

        # 打包本段结束时的参数值，作为复合输出
        final_transform_state_output = (
            translate_x_end, translate_y_end, scale_end,
            rotate_x_end, rotate_y_end, rotate_z_end,
            fixed_output_width, fixed_output_height
        )

        return (torch.cat(image_sequence, dim=0), torch.cat(mask_sequence, dim=0),
                final_transform_state_output)


    def _create_checkerboard_alpha_background(self, width, height, size=32):
        """创建用于默认图像的棋盘格，背景透明"""
        bg = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(bg)
        color_dark = (50, 50, 50, 255)
        color_light = (200, 200, 200, 255)

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

        if scale != 1.0:
            new_w = max(1, int(img.size[0] * scale))
            new_h = max(1, int(img.size[1] * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            original_alpha_pil = original_alpha_pil.resize((new_w, new_h), Image.LANCZOS) 

        rot_x, rot_y, rot_z = rotation_3d

        if rot_z != 0:
            fill_color = self._get_fill_color_tuple(bg_color, is_pil_fill=True)
            img = img.rotate(rot_z, expand=True, resample=Image.BICUBIC, fillcolor=fill_color)
            
            original_alpha_pil = original_alpha_pil.rotate(rot_z, expand=True, resample=Image.BICUBIC, fillcolor=0) 

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

        if external_background_pil:
            bg = external_background_pil
        else:
            bg = self._create_bg(w_orig_input_img, h_orig_input_img, bg_color)
        
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
            image = image.convert('RGB')
        
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def pil2mask_tensor(mask_image):
        """将 PIL 单通道图像（如Alpha通道）转换为 ComfyUI MASK Tensor"""
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "HOOTOO_ImageTransformSegment": HOOTOO_ImageTransformSegment
}

WEB_DIRECTORY = "./web"