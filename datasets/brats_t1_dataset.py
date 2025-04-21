import os
import re
import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms

class BraTST1Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.volume_dict = defaultdict(list)  # volume_x: [slice_paths]
        self.all_slices = []

        # 파일 스캔 및 volume 그룹핑
        for fname in os.listdir(data_dir):
            if not fname.endswith('.h5'):
                continue
            match = re.search(r'(volume_\d+)_slice_(\d+)\.h5$', fname)
            if match:
                vol_id = match.group(1)
                slice_idx = int(match.group(2))
                if 70 <= slice_idx <= 80:
                    path = os.path.join(data_dir, fname)
                    self.volume_dict[vol_id].append(path)
                    self.all_slices.append((vol_id, path))

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, idx):
        vol_id, path = self.all_slices[idx]

        with h5py.File(path, 'r') as f:
            t1 = f['image'][:, :, 0].astype('float32')  # (H, W)

        # Min-Max 정규화 (0~1 범위로)
        t1_min = t1.min()
        t1_max = t1.max()
        t1 = (t1 - t1_min) / (t1_max - t1_min)  # [0, 1] 범위로 변환

        # Tensor로 변환 후 차원 추가 (1, H, W)
        t1 = torch.tensor(t1, dtype=torch.float32).unsqueeze(0)

        # 추가적인 transform 적용 (이 경우 Normalize)
        if self.transform:
            t1 = self.transform(t1)

        return t1