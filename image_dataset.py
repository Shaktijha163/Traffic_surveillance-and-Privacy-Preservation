__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.1'
__status__ = "Research"
__date__ = "20/4/2018"
__license__= "MIT License"
import torch
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.stats import spearmanr



class ImageDataset(data.Dataset):

    def __init__(self, images, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.valid_labels = False

        self.data = images
        n = len(images)
        self.labels = np.zeros((n,), dtype=np.float32)

        return


    def __getitem__(self, index):
        sample = Image.fromarray(self.data[index])
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, str(index)


    def __len__(self):
        return len(self.data)

class MemCatVehicleDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, split_root, split, transform=None):
        self.images_dir = images_dir
        self.split_file = os.path.join(split_root, split)
        self.transform = transform
        
        # Load annotations
        annotations_file = os.path.join(os.path.dirname(split_root), 'annotations.txt')
        self.labels = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.labels[parts[0]] = float(parts[1])
        
        # Load image names from split file
        with open(self.split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        self.data = []
        for img_name in self.image_names:
            if img_name in self.labels:
                self.data.append((img_name, self.labels[img_name]))
            else:
                print(f"Warning: {img_name} not found in annotations")
        
        self.valid_labels = True
        print(f"Loaded {len(self.data)} samples from {split}")

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.images_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label, img_name

    def __len__(self):
        return len(self.data)

    def getRankCorrelationWithMSE(self, predictions, gt=None):
        """
        Calculate Spearman's rank correlation and MSE between predictions and ground truth
        """
        from scipy.stats import spearmanr
        
        if gt is None:
            # Use the labels from the dataset
            gt = [label for _, label in self.data]
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        gt = np.array(gt)
        
        # Calculate Spearman's rank correlation
        rc, _ = spearmanr(predictions, gt)
        
        # Calculate MSE
        mse = np.mean((predictions - gt) ** 2)
        
        return rc, mse
