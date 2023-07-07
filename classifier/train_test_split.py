import os
import shutil
import numpy as np 

train_dataset_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/train/'
val_dataset_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/val/'
test_dataset_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/test/'

n_labels = 9 
counts = [586,1631,517,66,267,164,6421,10632,698]

for i in range(n_labels):
    indices = list(range(counts[i]))
    train_split = int(np.floor(0.8 * counts[i]))
    val_split = train_split + int(np.floor(0.1 * counts[i]))
    np.random.shuffle(indices)
    train_folder = os.path.join(train_dataset_directory,str(i))
    val_folder = os.path.join(val_dataset_directory,str(i))
    test_folder = os.path.join(test_dataset_directory,str(i))
    count = 0
    for filename in sorted(os.listdir(train_folder)):
        if os.path.isfile(os.path.join(train_folder, filename)):
            if count in indices[train_split:val_split]:
                shutil.move(os.path.join(train_folder, filename), os.path.join(val_folder, filename))
            elif count in indices[val_split:]:
                shutil.move(os.path.join(train_folder, filename), os.path.join(test_folder, filename))
            count += 1
