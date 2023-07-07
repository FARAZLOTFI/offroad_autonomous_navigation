import os 
import numpy as np 
import cv2 

image_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/images/'
label_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/annotations/'
topic_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/topics/'
depth_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/depths/'
count = 0 
for filename in os.listdir(image_directory):
    print("Analyzing image: ", count, end='\r')
    n = len(filename)
    image = os.path.join(image_directory, filename)
    label = os.path.join(label_directory, filename.replace(filename[n - 3:], 'txt'))
    topic = os.path.join(topic_directory, filename.replace(filename[n - 3:], 'npy').replace(filename[:5], 'topics'))
    depth = os.path.join(depth_directory, filename.replace(filename[n - 3:], 'npy').replace(filename[:5], 'depth'))

    # checking for missing/corrupted files 
    if os.path.isfile(topic) and os.path.getsize(topic) > 0:
        with open(topic, 'rb') as datafile:
            data = np.load(datafile)
            if len(data) != 13:
                print("CORRUPTED TOPIC DATA FILE: " + topic)
                break
    else:
        print("MISSING/EMPTY TOPIC DATA FILE: " + topic)
        break

    if os.path.isfile(image) and os.path.getsize(image) > 0:
        im = cv2.imread(image)
        if im.shape != (480, 640, 3):
                print("CORRUPTED IMAGE DATA FILE: " + image)
    else:
        print("MISSING/EMPTY IMAGE FILE: " + image)
        break

    if os.path.isfile(depth) and os.path.getsize(depth) > 0:
        with open(depth, 'rb') as datafile:
            data = np.load(datafile)
            if data.shape != (480, 640):
                print("CORRUPTED DEPTH DATA FILE: " + depth)
                break
    else:
        print("MISSING/EMPTY DEPTH DATA FILE: " + depth)
        break

    if os.path.isfile(label) and os.path.getsize(label) > 0:
         with open(label) as file:
            l = int(file.readline())
            if l not in [0,1,2,3,4,5,6,7,8]:
                print("CORRUPTED LABEL DATA FILE: " + label)
                break
    else:
        print("MISSING/EMPTY LABEL DATA FILE: " + label)
        break

    count+=1

if count == 20982:
    print('Sanity check complete. No missing or corrupted files')



