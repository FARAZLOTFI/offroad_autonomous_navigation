import numpy as np
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

data_directory = '/home/farnoosh/offroad_autonomous_navigation/classifier/dataset'
np.random.seed(0)
batch_size = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
image_datasets = {x: datasets.ImageFolder(data_directory,
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}

N = len(image_datasets['train'])
indices = list(range(N))
train_split = int(np.floor(0.64 * N))
val_split = train_split + int(np.floor(0.16 * N))
np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(indices[:train_split])
val_sampler = SubsetRandomSampler(indices[train_split:val_split])
test_sampler = SubsetRandomSampler(indices[val_split:])

train_loader = torch.utils.data.DataLoader(image_datasets['train'], sampler=train_sampler, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(image_datasets['val'], sampler=val_sampler, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(image_datasets['test'], sampler=test_sampler, batch_size=64)

print(len(train_loader.dataset))