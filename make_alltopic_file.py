import os
import pickle

import numpy as np
from tqdm import tqdm

topic_path = '/home/nwaftp23/scratch/offroad_navigation_dataset/topics'
topics_files = os.listdir(topic_path)

all_topics = {}
for tf in tqdm(topics_files):
    all_topics[tf]=np.load(os.path.join(topic_path, tf))

with open(os.path.join(topic_path, 'all_topics.pkl'), 'wb') as f:
    pickle.dump(all_topics, f)
