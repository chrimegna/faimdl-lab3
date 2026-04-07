import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

def dataset(origine_train, origine_val):
  train_transform = transforms.Compose([
      transforms.RandomRotation(degrees=30),
      transforms.RandomHorizontalFlip(p=0.4),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  val_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) 
   
  train_dataset = ImageFolder(origine_train, transform=train_transform)
  val_dataset = ImageFolder(origine_val, transform=val_transform)
  return train_dataset, val_dataset

def dataloader(train_dataset, val_dataset, batch_size):
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
  return train_loader, val_loader
