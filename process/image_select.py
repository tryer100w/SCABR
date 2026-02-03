import os
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse


# 自定义数据集类
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


# 加载预训练的ResNet18模型
def load_model():
    # 使用 'weights' 参数替代 'pretrained'
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 去掉最后一层全连接层，用于特征提取
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


# 提取图片特征
def extract_features(model, dataloader):
    features = []
    file_names = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.extend(outputs.cpu().numpy())
            file_names.extend(names)
    return np.array(features), file_names


# 计算相似性得分
def calculate_similarity(features):
    similarity_matrix = cosine_similarity(features)
    # 只取矩阵的上三角部分（不包含对角线）
    upper_triangular = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
    # 计算平均相似性得分
    average_similarity = np.mean(upper_triangular)
    return average_similarity


# 主函数
def main(root_folder, folder1, folder2, folder5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_model()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    subfolder_scores = {}
    for sub_folder in os.listdir(root_folder):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            print(f"Processing sub-folder: {sub_folder_path}")
            dataset = ImageFolderDataset(sub_folder_path, transform=transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            features, _ = extract_features(model, dataloader)
            average_similarity = calculate_similarity(features)
            subfolder_scores[sub_folder] = average_similarity
            print(f"Sub-folder {sub_folder} similarity score: {average_similarity}")

    # 对得分进行排序
    sorted_scores = sorted(subfolder_scores.items(), key=lambda item: item[1], reverse=True)
    
    # 打印所有子文件夹的相似性得分
    print("\nAll sub-folder similarity scores:")
    for subfolder, score in sorted_scores:
        print(f"{subfolder}: {score}")
    
    top_20_count = int(len(sorted_scores) * 0.2)
    top_20_subfolders = [item[0] for item in sorted_scores[:top_20_count]]
    
    # 打印前20%的子文件夹
    print(f"\nTop 20% sub-folders (count: {top_20_count}):")
    for subfolder in top_20_subfolders:
        print(f"{subfolder}: {subfolder_scores[subfolder]}")

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(folder5):
        os.makedirs(folder5)
    
    # 遍历每个子文件夹
    for subfolder in subfolder_scores.keys():
        # 确定源文件夹
        source_folder = os.path.join(folder1, subfolder) if subfolder in top_20_subfolders else os.path.join(folder2, subfolder)
        
        if os.path.exists(source_folder):
            # 遍历源文件夹中的所有图片文件
            for file_name in os.listdir(source_folder):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_file = os.path.join(source_folder, file_name)
                    destination_file = os.path.join(folder5, file_name)
                    
                    # 复制文件（如果已存在则覆盖）
                    shutil.copy2(source_file, destination_file)
                    print(f"Copied {source_file} to {destination_file}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Select images based on similarity')
    parser.add_argument('--root_folder', type=str, required=True, help='Root folder path for calculating similarity')
    parser.add_argument('--folder1', type=str, required=True, help='Folder 1 path (high quality images)')
    parser.add_argument('--folder2', type=str, required=True, help='Folder 2 path (standard quality images)')
    parser.add_argument('--folder5', type=str, required=True, help='Output folder path')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数
    main(args.root_folder, args.folder1, args.folder2, args.folder5)
