import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class IAMProcessedDataset(Dataset):

    def __init__(self, lines_path, transform=None):

        self.transform = transform
        self.img_list = []
        for folder in os.listdir(lines_path):
            for inner_folder in os.listdir(os.path.join(lines_path, folder)):
                transcription_file = os.path.join(lines_path, folder, inner_folder, f'{inner_folder}-transcription.txt')
                with open(transcription_file, 'r') as f:
                    transcription = f.read()
                img_name = inner_folder
                img_path = os.path.join(lines_path, folder, inner_folder, f'{inner_folder}.png')
                self.img_list.append((img_name, img_path, transcription))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.img_list[idx][0]
        image_path = self.img_list[idx][1]
        transcription = self.img_list[idx][2]

        image = Image.open(image_path).convert("RGB")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        image = to_tensor(image)

        if self.transform:
            image = self.transform(image)

        sample = {'image_name': image_name, 'image': image, 'transcription': transcription}

        return sample
    
dataset = IAMProcessedDataset('lines_processed_padded')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print(len(dataset))
    
    

