HOOTOO_ImageTransform - 3D图像变换工具

HOOTOO_ImageTransform 是一个功能强大的3D图像变换工具节点，灵感来源于PT娃导。它能够对静态图像或默认生成的棋盘格进行复杂的动画变换，支持2D/3D变换、自定义背景和Mask输出。

✨ 核心特性
🎨 图像变换功能
2D/3D变换：支持平移、缩放和3D旋转(X/Y/Z轴)

透视效果：可调节焦距控制透视强度

动画生成：将单一图像生成为指定帧数的平滑动画序列

🖼️ 背景控制
自定义背景：支持外部图像/动画作为背景

内置选项：纯黑、纯白或棋盘格背景

无输入模式：自动生成大尺寸棋盘格作为操作对象

🎭 高级功能
Alpha通道支持：完整保留透明度信息

独立Mask输出：精确表示图像透明度，不含背景

固定输出尺寸：确保动画序列连贯性

🛠️ 接口说明
输入参数
参数名称	类型	说明
frames	int	要生成的动画帧数
output_width	int	输出图像宽度(无输入图像时生效)
output_height	int	输出图像高度(无输入图像时生效)
translate_x	float	X轴最终平移量
translate_y	float	Y轴最终平移量
scale_animate_start	float	起始帧缩放比例
scale_animate_end	float	结束帧缩放比例
rotate_x	float	X轴最终旋转角度(度)
rotate_y	float	Y轴最终旋转角度(度)
rotate_z	float	Z轴最终旋转角度(度)
focal_length	float	3D透视变换焦距
bg_color	enum	背景颜色("black", "white", "checker")
output_padding	float	图像内容与画布边缘的内边距
可选输入
background_image (IMAGE): 动画背景图像序列

输出
IMAGE (IMAGE): 变换后的RGBA图像序列

MASK (MASK): 纯Alpha通道序列

📌 使用提示
当不连接image输入时，节点会生成内部棋盘格

output_padding参数仅控制布局，不影响图像缩放

背景图像会自动缩放以适应输出尺寸

Mask输出完全透明(0值)的背景区域，便于合成



本节点和说明都由AI制作。灵感来源  PT 娃导

节点名称：HOOTOO_ImageTransform

概述：
HOOTOO_ImageTransform 节点是一个功能强大的 3D 图像变换工具，它允许您对静态图像或默认生成的棋盘格进行复杂的动画平移、缩放和 3D 旋转（X、Y、Z轴），并支持自定义背景和 Mask 输出。节点始终以固定尺寸输出结果，确保动画序列的连贯性。

核心特性：
2D/3D 变换: 支持图像在 2D 平面上的平移和缩放，以及围绕 X、Y、Z 轴的 3D 旋转，包括透视效果。
动画生成: 可将单一图像或默认棋盘格生成为指定帧数的动画序列，所有变换参数都将在这段动画中平滑过渡。

灵活的背景控制:
自定义背景图像: 可通过 background_image 接口传入外部图像或动画作为合成背景。
内置背景选项: 当没有外部背景时，可选择纯黑、纯白或棋盘格作为合成背景。
无输入图像模式: 如果不连接 image 输入，节点将生成一个内部的大尺寸棋盘格作为操作对象，并根据 output_width 和 output_height 参数输出固定尺寸的变换结果，实现“无限棋盘格”效果，避免缩小漏底。
Alpha 通道支持: 完全支持带有 Alpha 通道的图像输入，并在变换过程中保留透明度信息。
独立的 Mask 输出: 除了合成图像序列，还额外输出一个独立的 Mask 序列，精确表示图像内容的透明度信息，不含任何背景影响，便于后续的图像合成和处理。
固定输出尺寸: 无论输入图像尺寸如何，最终输出的图像和 Mask 序列都将保持一致的尺寸，确保动画的帧尺寸稳定。

接口说明：
frames : 要生成的动画帧数。
output_width : 最终输出图像的宽度。
output_height : 最终输出图像的高度。重要: 仅当 image 输入未连接时生效。如果 image 有连接，输出宽高同输出图像宽高。
translate_x : 图像在 X 轴上的最终平移量（第一帧为0，最后一帧为该值）。
translate_y : 图像在 Y 轴上的最终平移量（第一帧为0，最后一帧为该值）。
scale_animate_start : 动画第一帧的缩放比例。
scale_animate_end   : 动画最后一帧的缩放比例。
rotate_x : 图像在 X 轴上的最终旋转角度（度，第一帧为0，最后一帧为该值）。
rotate_y : 图像在 Y 轴上的最终旋转角度（度，第一帧为0，最后一帧为该值）。
rotate_z : 图像在 Z 轴上的最终旋转角度（度，第一帧为0，最后一帧为该值）。
focal_length : 3D 透视变换的焦距。值越大，透视效果越弱；值越小，透视效果越强。
bg_color (选择: "black", "white", "checker", 默认: "black"): 当没有 background_image 连接时，可选择checker用作图像合成的背景输出。
注意：在有图像输入但没有 background_image 输入的情况下，如果 bg_color 选择 "checker"，旋转产生的透明区域会显示出棋盘格。

output_padding : 图像内容与最终输出画布边缘之间的内边距。图像内容会在此边距定义的内部区域中居中。设置为 0 则没有额外边距。提示: 此参数仅用于布局控制，不会强制缩小放大后的图像。
background_image (可选, IMAGE): 动画背景图像序列。
如果连接此接口，则其每一帧将用作对应输出帧的背景。
如果 background_image 的帧数与 frames 参数不匹配，节点会循环使用 background_image 的帧。
如果其尺寸与最终输出尺寸不匹配，它将被自动缩放以适应。
如果未连接此接口，背景将由 bg_color 参数决定。

输出接口 (Outputs)：
IMAGE (IMAGE): 经过 3D 变换并与背景合成后的图像序列。该序列保留了 Alpha 通道信息 (RGBA 格式)。
MASK (MASK): 经过 3D 变换后的纯 Alpha 通道序列。该 Mask 仅反映前景图像内容的透明度，背景区域在 Mask 上表现为完全透明（0 值），方便与其他 Mask 操作或合成节点配合使用。
<img width="1589" alt="1" src="https://github.com/user-attachments/assets/3292ae57-779e-41b2-b03c-55c8127de0b0" />

https://github.com/user-attachments/assets/4b267a4e-91a0-452a-81f0-04e4d28dd417   

https://github.com/user-attachments/assets/53d08339-a055-40ff-8092-31488e2ee27f

https://github.com/user-attachments/assets/30d7daa4-0793-473d-ab52-9b8dc4a6fc8d

https://github.com/user-attachments/assets/5f5bac22-fb04-46bd-93d9-55a71af3c2e9

https://github.com/user-attachments/assets/837f2982-7314-47db-a9a7-a6423cdc2418



