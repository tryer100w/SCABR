import torchvision


# train_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/base',
#         'tieredImageNet': 'dataset/tieredImageNet/base',
#         'CIFAR-FS': 'dataset/cifar100/base',
#         'FC100': 'dataset/FC100_hd/base',
#     }
#
# val_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/val',
#         'tieredImageNet': 'dataset/tieredImageNet/val',
#         'CIFAR-FS': 'dataset/cifar100/val',
#         'FC100': 'dataset/FC100_hd/val',
#     }
#
# test_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/novel',
#         'tieredImageNet': 'dataset/tieredImageNet/novel',
#         'CIFAR-FS': 'dataset/cifar100/novel',
#         'FC100': 'dataset/FC100_hd/novel',
#     }
#
# dataset_path = {
#         'miniImageNet': ['dataset/miniImagenet/base', 'dataset/miniImagenet/val', 'dataset/miniImagenet/novel'],
#         'tieredImageNet': ['dataset/tieredImageNet/base', 'dataset/tieredImageNet/val', 'dataset/tieredImageNet/novel'],
#         'CIFAR-FS': ['dataset/cifar100/base', 'dataset/cifar100/val', 'dataset/cifar100/novel'],
#         'FC100': ['dataset/FC100_hd/base', 'dataset/FC100_hd/val', 'dataset/FC100_hd/novel'],
# }

train_dataset_path = {
        'miniImageNet': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/mini/train',
        'tieredImageNet': '/share/home/ncu_418000240017/FeatWalk_copy/filelist/tieredImageNet/train5',
        'CUB': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/base_fore_vst',
        'CIFAR-FS': '/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/base',
        'FC100': '/home/wht/Code/FeatWalk_copy/filelist/FC100/base',
        'FC100_hd': '/home/wht/Code/FeatWalk_copy/filelist/FC100_hd/base',
        'CropDisease': '/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/CropDisease/base'
    }

val_dataset_path = {
        'miniImageNet': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/mini/val',
        'tieredImageNet': '/share/home/ncu_418000240017/FeatWalk_copy/filelist/tieredImageNet/val5',
        'CUB': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/val_fore_vst',
        'CIFAR-FS': '/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/val',
        'FC100': '/home/wht/Code/FeatWalk_copy/filelist/FC100/val',
        'CropDisease': '/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/CropDisease/val'
    }

test_dataset_path = {
        'miniImageNet': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/mini/test_cluster_select',
        'tieredImageNet': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/tieredImageNet/test5',
        'CUB': '/public/home/ncu_418000240017/FeatWalk_copy/filelist/CUB/novel_fore_vst',
        'CIFAR-FS': '/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/novel',
        'FC100': '/home/wht/Code/FeatWalk_copy/filelist/FC100/novel',
        'FC100_hd': '/home/wht/Code/FeatWalk_copy/filelist/FC100_hd/novel',
        'EuroSAT': '/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/EuroSAT/2750',
        'CropDisease': '/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/CropDisease/novel'
    }



dataset_path = {
        'miniImageNet': ['/home/wht/Code/FeatWalk_copy/filelist/miniImagenet/base5', '/home/wht/Code/FeatWalk_copy/filelist/miniImagenet/val5', '/home/wht/Code/FeatWalk_copy/filelist/miniImageNet/novel5'],
        'tieredImageNet': ['/home/wht/Code/FeatWalk_copy/filelist/tieredImageNet/base5', '/home/wht/Code/FeatWalk_copy/filelist/tieredImageNet/val5', '/home/wht/Code/FeatWalk_copy/filelist/tieredImageNet/novel5'],
        'CUB': ['/home/wht/Code/FeatWalk_copy/filelist/CUB/base', '/home/wht/Code/FeatWalk_copy/filelist/CUB/val', '/home/wht/Code/FeatWalk_copy/filelist/CUB/novel'],
        'CIFAR-FS': ['/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/base', '/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/val', '/home/wht/Code/FeatWalk_copy/filelist/CIFAR-FS/cifar100/novel'],
        'FC100': ['/home/wht/Code/FeatWalk_copy/filelist/FC100/base', '/home/wht/Code/FeatWalk_copy/filelist/FC100/val', '/home/wht/Code/FeatWalk_copy/filelist/FC100/novel'],
        'CropDisease': ['/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/CropDisease/train'],
        'EuroSAT': ['/home/wht/Code/FeatWalk_copy/filelist/CD-FSL/EuroSAT/2750']
}


class DatasetWithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        if split == 'train':
            dataset_path = train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = test_dataset_path[dataset_name]
        self.dataset = torchvision.datasets.ImageFolder(dataset_path, aug)
        self.idx2text = {}
        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('data/features.txt', 'r') as f:
                for line in f.readlines():
                    # 原代码：split() 会按所有空格分割，导致文本含空格时元素数量超过3
                    # idx, _, text = line.strip().split()
                    
                    # 修改为：maxsplit=2 仅分割前两个空格，保留后续文本为整体
                    parts = line.strip().split(maxsplit=2)
                    if len(parts) < 3:  # 跳过格式不完整的行（避免报错）
                        continue
                    idx, _, text = parts  # 正确 unpack 为3个元素
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'CUB':
            # 尝试读取CUB的详细描述文件
            cub_description_path = 'data/CUB.txt'
            try:
                with open(cub_description_path, 'r') as f:
                    for line in f.readlines():
                        # 假设格式为：001.Black_footed_Albatross: 详细描述文本
                        line = line.strip()
                        if ':' in line:
                            idx, description = line.split(':', 1)  # 只分割第一个冒号
                            idx = idx.strip()
                            description = description.strip().replace('_', ' ')
                            self.idx2text[idx] = description
                            
                # 确保所有类别都有对应的文本描述
                for idx in self.dataset.classes:
                    if idx not in self.idx2text:
                        # 如果在描述文件中找不到该类别，则使用原始逻辑
                        if len(idx) > 4 and idx[3] == '.':
                            processed_idx = idx[4:]
                        else:
                            processed_idx = idx
                        text = processed_idx.replace('_', ' ')
                        self.idx2text[idx] = text
            except FileNotFoundError:
                # 如果没有找到详细描述文件，则使用原始逻辑
                for idx in self.dataset.classes:
                    # 去掉前四位（如001.，178.）
                    if len(idx) > 4 and idx[3] == '.':
                        processed_idx = idx[4:]
                    else:
                        processed_idx = idx
                    # 将下划线替换为空格
                    text = processed_idx.replace('_', ' ')
                    self.idx2text[idx] = text

    def __getitem__(self, i):
        image, label = self.dataset[i]
        text = self.dataset.classes[label]
        text = self.idx2text[text]
        # text prompt: A photo of a {label}
        text = 'A photo of a ' + text
        return image, label, text

    def __len__(self):
        return len(self.dataset)
