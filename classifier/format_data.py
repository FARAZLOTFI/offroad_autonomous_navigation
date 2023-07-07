import os
import shutil

image_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/images/'
label_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/annotations/'
dataset_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/train/'

for filename in os.listdir(image_directory):
    f = os.path.join(label_directory, filename.replace(filename[len(filename) - 3:], 'txt'))
    # checking if it is a file
    if os.path.isfile(f):
        with open(f) as file:
            label = file.readline() + '/'
            copy_path = os.path.join(dataset_directory, label)
            shutil.copy(os.path.join(image_directory, filename), os.path.join(copy_path, filename))
