import os
import numpy as np
from PIL import Image

def remove_background(image_path, saliency_path):
    # 打开图片和显著图
    image = Image.open(image_path).convert("RGBA")
    saliency = Image.open(saliency_path).convert("L")
    
    # 将显著图转换为numpy数组
    saliency_array = np.array(saliency)
    
    # 创建一个新的alpha通道
    alpha = np.where(saliency_array > 128, 255, 0).astype(np.uint8)
    
    # 将alpha通道添加到图片中
    image.putalpha(Image.fromarray(alpha))
    
    # 获取图像的RGB通道
    r, g, b, a = image.split()
    
    # 创建一个白色背景图像
    white_background = Image.new("RGB", image.size, (255, 255, 255))
    
    # 将图像与白色背景合并得到前景图
    foreground = Image.composite(image.convert("RGB"), white_background, a)
    
    # 创建反向alpha通道以提取背景
    inv_alpha = np.where(saliency_array > 128, 0, 255).astype(np.uint8)
    inv_alpha_img = Image.fromarray(inv_alpha)
    
    # 使用反向alpha通道提取背景
    background = Image.composite(image.convert("RGB"), white_background, inv_alpha_img)
    
    return foreground, background

def process_folders(image_folder, saliency_folder, foreground_output_folder, background_output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(foreground_output_folder):
        os.makedirs(foreground_output_folder)
    if not os.path.exists(background_output_folder):
        os.makedirs(background_output_folder)
    
    # 遍历文件夹中的文件
    for filename in os.listdir(image_folder):
        # 修改: 支持jpg和png格式的文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            saliency_filename = os.path.splitext(filename)[0] + '.png'
            saliency_path = os.path.join(saliency_folder, saliency_filename)
            
            if os.path.exists(saliency_path):
                # 去除背景，获取前景和背景
                foreground, background = remove_background(image_path, saliency_path)
                
                # 保存前景图片
                foreground_path = os.path.join(foreground_output_folder, filename)
                # 修改: 将图像模式从RGBA转换为RGB以支持JPEG格式
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    foreground = foreground.convert("RGB")
                foreground.save(foreground_path)
                
                # 保存背景图片
                background_path = os.path.join(background_output_folder, filename)
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    background = background.convert("RGB")
                background.save(background_path)
                
                print(f"Processed {filename} - saved foreground to {foreground_path} and background to {background_path}")
            else:
                print(f"Saliency map not found for {filename}")

if __name__ == "__main__":
    # 指定源文件夹路径和输出文件夹路径
    image_folder = "/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/base"
    saliency_folder = "/public/home/ncu_418000240017/XRH/RGB_VST/preds/CUB/base"
    foreground_output_folder = "/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/base_fore_vst"
    background_output_folder = "/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/base_back_vst"
    process_folders(image_folder, saliency_folder, foreground_output_folder, background_output_folder)
