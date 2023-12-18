import os
import torch
from torch.utils.data import Dataset

class PtFolderDataset(Dataset):
    def __init__(self, root):

        self.data = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            assert os.path.isdir(class_dir)

            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            for file_name in os.listdir(class_dir):
                assert file_name.endswith('.pt')

                file_path = os.path.join(class_dir, file_name)
                self.data.append((file_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, class_idx = self.data[idx]
        x = torch.load(file_path)
        x = torch.squeeze(x, 0)
        return x, class_idx

            