import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.semi_supervised import LabelPropagation
from PIL import Image
import torchvision.transforms as transforms
import os
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn



class RepeatSampler:
    def __init__(self, dataset, batch_size, repeat):
        self.batch_size = batch_size//repeat
        self.repeat = repeat
        self.sampler = torch.utils.data.RandomSampler(dataset)
        self.drop_last = True

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = batch * self.repeat
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class MultiTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        out = []
        for trans in self.trans:
            out.append(trans(x))
        return out


class U2NetProcessor:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        if model_path is None:
            raise ValueError("U2Net model path must be provided. U2Net is required and cannot be skipped.")
        
        if not os.path.exists(model_path):
            print(f"U2Net model file not found at {model_path}. Attempting to download...")
            self._download_u2net_model(model_path)
        
        try:
            loaded_data = torch.load(model_path, map_location=self.device)
            
            if isinstance(loaded_data, dict):
                if 'state_dict' in loaded_data:
                    state_dict = loaded_data['state_dict']
                else:
                    state_dict = loaded_data
                
                self.model = self._create_u2net_model()
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"U2Net model successfully loaded from {model_path}")
            else:
                self.model = loaded_data
                self.model.eval()
                print(f"U2Net model successfully loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load U2Net model from {model_path}: {e}. U2Net is required and cannot be skipped.")
    
    def _download_u2net_model(self, model_path):
        """
        Download the U2Net model from the official repository.
        
        Args:
            model_path (str): Path to save the downloaded model.
        
        Raises:
            RuntimeError: If download fails or the downloaded file is incomplete.
        """
        import urllib.request
        import os
        
        # URL of the pre-trained U2Net model
        url = "https://github.com/NathanUA/U-2-Net/releases/download/1.0.0/u2net.pth"
        
        # Expected file size in bytes (approx. 165MB)
        expected_size = 173015040
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print(f"Downloading U2Net model from {url} to {model_path} (approx. 165MB)...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            
            # Check if the downloaded file has the expected size
            actual_size = os.path.getsize(model_path)
            if abs(actual_size - expected_size) > 1024 * 1024:  # Allow 1MB difference
                os.remove(model_path)  # Remove incomplete file
                raise RuntimeError(f"Downloaded file size ({actual_size} bytes) does not match expected size ({expected_size} bytes). The file may be incomplete.")
            
            print(f"U2Net model successfully downloaded to {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download U2Net model from {url}: {e}")
    
    def _create_u2net_model(self):
        class U2NET(nn.Module):
            def __init__(self, in_ch=3, out_ch=1):
                super(U2NET, self).__init__()
                
                self.encoder1 = RSU7(in_ch, 32, 64)
                self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
                
                self.encoder2 = RSU6(64, 32, 128)
                self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
                
                self.encoder3 = RSU5(128, 64, 256)
                self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
                
                self.encoder4 = RSU4(256, 128, 512)
                self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
                
                self.encoder5 = RSU4F(512, 256, 512)
                self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)
                
                self.encoder6 = RSU4F(512, 256, 512)
                
                self.decoder6 = RSU4F(512, 256, 512)
                
                self.decoder5 = RSU4F(1024, 256, 512)
                self.decoder4 = RSU4(1024, 256, 256)
                self.decoder3 = RSU5(512, 128, 128)
                self.decoder2 = RSU6(256, 64, 64)
                self.decoder1 = RSU7(128, 64, 64)
                
                self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
                self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
                self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
                self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
                self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
                self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
                
                self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
            
            def forward(self, x):
                h1 = self.encoder1(x)
                h = self.pool1(h1)
                
                h2 = self.encoder2(h)
                h = self.pool2(h2)
                
                h3 = self.encoder3(h)
                h = self.pool3(h3)
                
                h4 = self.encoder4(h)
                h = self.pool4(h4)
                
                h5 = self.encoder5(h)
                h = self.pool5(h5)
                
                h6 = self.encoder6(h)
                
                d6 = self.decoder6(h6)
                d6 = F.interpolate(d6, scale_factor=2, mode='bilinear', align_corners=False)
                
                d5 = self.decoder5(torch.cat((d6, h5), 1))
                d5 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=False)
                
                d4 = self.decoder4(torch.cat((d5, h4), 1))
                d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
                
                d3 = self.decoder3(torch.cat((d4, h3), 1))
                d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
                
                d2 = self.decoder2(torch.cat((d3, h2), 1))
                d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
                
                d1 = self.decoder1(torch.cat((d2, h1), 1))
                
                side6 = self.side6(d6)
                side5 = self.side5(d5)
                side4 = self.side4(d4)
                side3 = self.side3(d3)
                side2 = self.side2(d2)
                side1 = self.side1(d1)
                
                side6 = F.interpolate(side6, scale_factor=64, mode='bilinear', align_corners=False)
                side5 = F.interpolate(side5, scale_factor=32, mode='bilinear', align_corners=False)
                side4 = F.interpolate(side4, scale_factor=16, mode='bilinear', align_corners=False)
                side3 = F.interpolate(side3, scale_factor=8, mode='bilinear', align_corners=False)
                side2 = F.interpolate(side2, scale_factor=4, mode='bilinear', align_corners=False)
                side1 = F.interpolate(side1, scale_factor=2, mode='bilinear', align_corners=False)
                
                out = torch.cat([side1, side2, side3, side4, side5, side6], dim=1)
                out = self.outconv(out)
                
                return [out]
        
        class RSU7(nn.Module):
            def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
                super(RSU7, self).__init__()
                
                self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
                self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
                self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
                
                self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
            
            def forward(self, x):
                hx = x
                hxin = self.rebnconvin(hx)
                
                hx1 = self.rebnconv1(hxin)
                hx = self.pool1(hx1)
                
                hx2 = self.rebnconv2(hx)
                hx = self.pool2(hx2)
                
                hx3 = self.rebnconv3(hx)
                hx = self.pool3(hx3)
                
                hx4 = self.rebnconv4(hx)
                hx = self.pool4(hx4)
                
                hx5 = self.rebnconv5(hx)
                hx = self.pool5(hx5)
                
                hx6 = self.rebnconv6(hx)
                
                hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
                hx5dup = F.interpolate(hx5d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
                hx4dup = F.interpolate(hx4d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
                hx3dup = F.interpolate(hx3d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
                hx2dup = F.interpolate(hx2d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
                
                return hx1d + hxin
        
        class RSU6(nn.Module):
            def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
                super(RSU6, self).__init__()
                
                self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
                self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
                self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
                
                self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
            
            def forward(self, x):
                hx = x
                hxin = self.rebnconvin(hx)
                
                hx1 = self.rebnconv1(hxin)
                hx = self.pool1(hx1)
                
                hx2 = self.rebnconv2(hx)
                hx = self.pool2(hx2)
                
                hx3 = self.rebnconv3(hx)
                hx = self.pool3(hx3)
                
                hx4 = self.rebnconv4(hx)
                hx = self.pool4(hx4)
                
                hx5 = self.rebnconv5(hx)
                
                hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
                hx4dup = F.interpolate(hx4d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
                hx3dup = F.interpolate(hx3d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
                hx2dup = F.interpolate(hx2d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
                
                return hx1d + hxin
        
        class RSU5(nn.Module):
            def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
                super(RSU5, self).__init__()
                
                self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
                self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
                self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
                
                self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
            
            def forward(self, x):
                hx = x
                hxin = self.rebnconvin(hx)
                
                hx1 = self.rebnconv1(hxin)
                hx = self.pool1(hx1)
                
                hx2 = self.rebnconv2(hx)
                hx = self.pool2(hx2)
                
                hx3 = self.rebnconv3(hx)
                hx = self.pool3(hx3)
                
                hx4 = self.rebnconv4(hx)
                
                hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
                hx3dup = F.interpolate(hx3d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
                hx2dup = F.interpolate(hx2d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
                
                return hx1d + hxin
        
        class RSU4(nn.Module):
            def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
                super(RSU4, self).__init__()
                
                self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
                self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
                self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
                self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
                
                self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
                
                self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
                self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
            
            def forward(self, x):
                hx = x
                hxin = self.rebnconvin(hx)
                
                hx1 = self.rebnconv1(hxin)
                hx = self.pool1(hx1)
                
                hx2 = self.rebnconv2(hx)
                hx = self.pool2(hx2)
                
                hx3 = self.rebnconv3(hx)
                
                hx2d = self.rebnconv2d(torch.cat((hx3, hx2), 1))
                hx2dup = F.interpolate(hx2d, scale_factor=2, mode='bilinear', align_corners=False)
                
                hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
                
                return hx1d + hxin
        
        class RSU4F(nn.Module):
            def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
                super(RSU4F, self).__init__()
                
                self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
                self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
                self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
                self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
                
                self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
                self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
                self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
            
            def forward(self, x):
                hx = x
                hxin = self.rebnconvin(hx)
                
                hx1 = self.rebnconv1(hxin)
                hx2 = self.rebnconv2(hx1)
                hx3 = self.rebnconv3(hx2)
                
                hx3d = self.rebnconv3d(torch.cat((hx3, hx2), 1))
                hx2d = self.rebnconv2d(torch.cat((hx3d, hx1), 1))
                hx1d = self.rebnconv1d(torch.cat((hx2d, hxin), 1))
                
                return hx1d + hxin
        
        class REBNCONV(nn.Module):
            def __init__(self, in_ch=3, out_ch=3, dirate=1):
                super(REBNCONV, self).__init__()
                
                # 修改属性命名以匹配预训练模型
                self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=dirate)
                self.bn_s1 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                # 使用新的属性名
                return self.relu(self.bn_s1(self.conv_s1(x)))
        
        return U2NET(in_ch=3, out_ch=1).to(self.device)
    
    def extract_foreground(self, image):
        if self.model is None:
            raise RuntimeError("U2Net model is not loaded. Cannot extract foreground.")
        
        try:
            with torch.no_grad():
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                outputs = self.model(input_tensor)
                mask = outputs[0][:, 0, :, :]
                mask = torch.sigmoid(mask)
                mask = (mask > 0.5).float()
                mask = F.interpolate(mask.unsqueeze(1), size=image.size[::-1], mode='bilinear', align_corners=False)
                mask = mask.squeeze().cpu().numpy()
            return mask
        except Exception as e:
            raise RuntimeError(f"Failed to extract foreground using U2Net: {e}")
    
    def remove_background(self, image, mask):
        if mask is None:
            raise ValueError("Mask cannot be None. U2Net foreground extraction is required.")
        
        image_array = np.array(image)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = image_array * mask_3d + 255 * (1 - mask_3d)
        return Image.fromarray(result.astype(np.uint8))


class LabelPropagator:
    def __init__(self, n_classes=5, kernel='rbf', gamma=20):
        self.n_classes = n_classes
        self.kernel = kernel
        self.gamma = gamma
    
    def propagate_labels(self, features, labeled_indices, true_labels, unlabeled_indices):
        n_samples = len(features)
        labels = np.full(n_samples, -1)
        labels[labeled_indices] = true_labels
        
        lp = LabelPropagation(kernel=self.kernel, gamma=self.gamma, n_jobs=-1)
        lp.fit(features, labels)
        
        predicted_labels = lp.transduction_
        return predicted_labels
    
    def extract_features(self, images):
        features = []
        for img in images:
            if isinstance(img, Image.Image):
                img_array = np.array(img.resize((224, 224)))
            else:
                img_array = np.array(img)
            if len(img_array.shape) == 3:
                features.append(img_array.flatten())
            else:
                features.append(img_array.flatten())
        return np.array(features)


class BackgroundSimilarityCalculator:
    def __init__(self, threshold=0.68):
        self.threshold = threshold
    
    def calculate_background_similarity(self, images, masks):
        if len(images) < 2:
            return 1.0
        
        background_features = []
        for img, mask in zip(images, masks):
            if mask is None:
                bg_feature = self._extract_feature(img)
            else:
                bg_image = self._extract_background(img, mask)
                bg_feature = self._extract_feature(bg_image)
            background_features.append(bg_feature)
        
        similarities = []
        n = len(background_features)
        for i in range(n):
            for j in range(i + 1, n):
                sim = 1 - cosine(background_features[i], background_features[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        return avg_similarity
    
    def _extract_background(self, image, mask):
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        mask_3d = np.stack([1 - mask] * 3, axis=-1)
        background = img_array * mask_3d
        return background
    
    def _extract_feature(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image.resize((64, 64)))
        else:
            if len(image.shape) == 3:
                img_array = Image.fromarray(image)
                img_array = np.array(img_array.resize((64, 64)))
            else:
                img_array = image
        
        if len(img_array.shape) == 3:
            return img_array.flatten() / 255.0
        else:
            return img_array.flatten() / 255.0
    
    def should_keep_original(self, similarity):
        return similarity >= self.threshold


class EpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True, 
                 u2net_model_path=None, similarity_threshold=0.68, device='cuda'):
        if u2net_model_path is None:
            raise ValueError("u2net_model_path must be provided. U2Net is required and cannot be skipped.")
        
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed
        
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
                self.cached_batches.append(batch)
            self.cached_batches = torch.stack(self.cached_batches)
            np.random.seed(0)
        
        self.u2net_processor = U2NetProcessor(model_path=u2net_model_path, device=device)
        self.label_propagator = LabelPropagator(n_classes=5)
        self.similarity_calculator = BackgroundSimilarityCalculator(threshold=similarity_threshold)
        self.processed_cache = {}
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed
        
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
                self.cached_batches.append(batch)
            self.cached_batches = torch.stack(self.cached_batches)
            np.random.seed(0)
        
        self.u2net_processor = U2NetProcessor(model_path=u2net_model_path, device=device)
        self.label_propagator = LabelPropagator(n_classes=5)
        self.similarity_calculator = BackgroundSimilarityCalculator(threshold=similarity_threshold)
        self.processed_cache = {}
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            if self.fix_seed:
                batch_indices = self.cached_batches[i_batch]
            else:
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch_indices = torch.stack(batch).reshape(-1)
            
            yield batch_indices
    
    def process_episode(self, dataset, batch_indices, return_images=False):
        cache_key = tuple(batch_indices.tolist())
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        images = []
        for idx in batch_indices:
            img, _ = dataset[idx]
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            images.append(img)
        
        labeled_indices = list(range(self.n_cls))
        true_labels = list(range(self.n_cls))
        unlabeled_indices = list(range(self.n_cls, len(images)))
        
        features = self.label_propagator.extract_features(images)
        predicted_labels = self.label_propagator.propagate_labels(
            features, labeled_indices, true_labels, unlabeled_indices
        )
        
        processed_images = []
        class_groups = {}
        for i, label in enumerate(predicted_labels):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(images[i])
        
        for label in class_groups:
            class_images = class_groups[label]
            masks = []
            for img in class_images:
                mask = self.u2net_processor.extract_foreground(img)
                masks.append(mask)
            
            similarity = self.similarity_calculator.calculate_background_similarity(class_images, masks)
            
            if self.similarity_calculator.should_keep_original(similarity):
                processed_images.extend(class_images)
            else:
                for img, mask in zip(class_images, masks):
                    processed_img = self.u2net_processor.remove_background(img, mask)
                    processed_images.append(processed_img)
        
        result = {'images': processed_images, 'labels': predicted_labels, 'indices': batch_indices}
        self.processed_cache[cache_key] = result
        
        if return_images:
            return result
        else:
            return batch_indices
    
    def clear_cache(self):
        self.processed_cache.clear()
