# datasets.py
import random
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnalignedDataset(Dataset):
    """
    Loads unpaired images from trainA and trainB folders (CycleGAN style).
    """
    def __init__(self, root, phase='train', load_size=286, crop_size=256):
        # expected root structure: root/trainA, root/trainB, root/testA, root/testB
        self.dir_A = os.path.join(root, f"{phase}A")
        self.dir_B = os.path.join(root, f"{phase}B")
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.size_A = len(self.A_paths)
        self.size_B = len(self.B_paths)

        transform_list = [
            transforms.Resize((load_size, load_size), interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.size_A]
        B_index = random.randint(0, self.size_B - 1)
        B_path = self.B_paths[B_index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.size_A, self.size_B)
