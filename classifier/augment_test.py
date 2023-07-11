
import os
from PIL import Image
import random 
from torchvision import transforms

test_folder = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/test/'
num_classes = 9 
num_augmentations_per_class = 100

augment = transforms.Compose([ 
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4, 0.3, 0.3, 0.3),      
    transforms.GaussianBlur(kernel_size=(9,9)),
    ])

for i in range(num_classes):
    data_folder = os.path.join(test_folder, str(i))
    for j in range(num_augmentations_per_class):
        image_file = os.path.join(data_folder,random.choice(os.listdir(data_folder)))
        image = Image.open(image_file)
        image = augment(image)
        image.save(os.path.join(data_folder, 'augmented_{}_{}.jpg'.format(i,j)))
