import os
import shutil
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # 新增：轮廓系数计算
from tqdm import tqdm  # 进度条可视化
import argparse  # 新增：命令行参数解析


def is_image_file(filename):
    """判断是否为图片文件（复用常见图片格式判断）"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    return os.path.splitext(filename)[1].lower() in valid_ext

def extract_features(img_path, model, transform, device):  # 新增device参数
    """使用预训练模型提取图片特征（修复设备不匹配问题）"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # 转换为[1, 3, 224, 224]的张量（CPU）
        img = img.to(device)  # 关键修改：将输入张量同步到模型所在设备（GPU/CPU）
        with torch.no_grad():
            features = model(img)
        return features.squeeze().cpu().numpy()  # 结果转回CPU并转为numpy
    except Exception as e:
        print(f"提取特征失败 {img_path}: {str(e)}")
        return None

def cluster_images(input_folder, output_root, n_clusters=None, max_clusters=23, model_name='resnet18'):
    """
    对文件夹内的图片进行聚类（支持自适应类别数量）
    :param n_clusters: 手动指定聚类数（若为None则自动选择）
    :param max_clusters: 自适应模式下最大尝试的聚类数（默认15）
    """
    # 步骤1：初始化模型并获取设备信息（关键修改）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 新增设备定义
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model = model.features
    else:
        raise ValueError(f"不支持的模型 {model_name}")
    model.eval()
    model = model.to(device)  # 模型移动到目标设备

    # 步骤2：预处理（未修改）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 步骤3：提取特征（修改调用方式）
    image_paths = []
    features_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if is_image_file(file):
                img_path = os.path.join(root, file)
                # 关键修改：传递device参数到特征提取函数
                feature = extract_features(img_path, model, transform, device)
                if feature is not None:
                    image_paths.append(img_path)
                    features_list.append(feature)
    
    if len(features_list) == 0:
        print("未找到可处理的图片文件")
        return
    features = np.array(features_list)

    # 新增：自适应选择最佳聚类数（当n_clusters为None时触发）
    if n_clusters is None:
        best_k = 10       #默认最佳聚类数
        best_score = -1  # 轮廓系数范围[-1,1]，初始设为-1
        # 遍历k=2到max_clusters（k=1无意义）
        for k in tqdm(range(10, max_clusters+1), desc="寻找最佳聚类数"):
            if k > len(features):  # 聚类数不能超过样本数
                break
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            # 计算轮廓系数（样本n_clusters数需>k）
            if len(features) > k:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        n_clusters = best_k
        print(f"自适应选择最佳聚类数：{n_clusters}（轮廓系数：{best_score:.4f}）")

    # 步骤4：聚类（未修改，使用确定的n_clusters）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # 获取输入文件夹的最后一级目录名
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    # 步骤5：按聚类结果复制图片到对应子文件夹
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        # 为每个聚类创建子文件夹（格式：cluster_0, cluster_1...）
        #cluster_folder = os.path.join(output_root, f"cluster_{label}")
        cluster_folder = os.path.join(output_root, f"cluster_{input_folder_name}_{label}")
        os.makedirs(cluster_folder, exist_ok=True)  # 自动创建目录（若不存在）
        
        # 构建目标路径并复制图片
        target_path = os.path.join(cluster_folder, os.path.basename(img_path))
        shutil.copy2(img_path, target_path)  # 复制原文件到目标簇文件夹
    
    print(f"聚类完成！共处理 {len(image_paths)} 张图片，分类到 {n_clusters} 个聚类中")

if __name__ == "__main__":
    # 新增：命令行参数解析
    parser = argparse.ArgumentParser(description='图片聚类工具')
    parser.add_argument('--input_folder', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output_root', type=str, required=True, help='聚类结果输出根目录')
    parser.add_argument('--n_clusters', type=int, default=None, help='聚类数量（默认为None，自动选择）')
    parser.add_argument('--max_clusters', type=int, default=23, help='自动选择时最大尝试的聚类数（默认15）')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'vgg16'], help='特征提取模型（默认resnet18）')
    
    args = parser.parse_args()
    
    # 调用聚类函数
    cluster_images(
        input_folder=args.input_folder,
        output_root=args.output_root,
        n_clusters=args.n_clusters,
        max_clusters=args.max_clusters,
        model_name=args.model_name
    )
