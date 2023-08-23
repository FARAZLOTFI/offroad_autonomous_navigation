import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator


path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/validation"
filename = "events.out.tfevents.1692731716.bg12112.int.ets1.calculquebec.ca.248769.1"
filename = os.path.join(path, filename)
total_val_loss = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        total_val_loss.append(loss)
    
path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/val_loss_term0"
filename = "events.out.tfevents.1692731719.bg12112.int.ets1.calculquebec.ca.248769.3"
filename = os.path.join(path, filename)
val_loss_0 = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        val_loss_0.append(loss)

path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/val_loss_term1"
filename = "events.out.tfevents.1692731733.bg12112.int.ets1.calculquebec.ca.248769.5"
filename = os.path.join(path, filename)
val_loss_1 = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        val_loss_1.append(loss)

path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/training"
filename = "events.out.tfevents.1692731711.bg12112.int.ets1.calculquebec.ca.248769.0"
filename = os.path.join(path, filename)
total_train_loss = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        total_train_loss.append(loss)
    
path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/train_loss_term0"
filename = "events.out.tfevents.1692731718.bg12112.int.ets1.calculquebec.ca.248769.2"
filename = os.path.join(path, filename)
train_loss_0 = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        train_loss_0.append(loss)

path = "/home/nwaftp23/scratch/offroad_trainfiles/traj_planner_training_logs/runs/val_loss_term1"
filename = "events.out.tfevents.1692731733.bg12112.int.ets1.calculquebec.ca.248769.5"
filename = os.path.join(path, filename)
train_loss_1 = []
for i, summary in enumerate(summary_iterator(filename)):
    if i>1:
        loss = summary.summary.value._values[0].simple_value
        train_loss_1.append(loss)

fig, ax = plt.subplots(1,2,figsize=(10,5))
x = np.arange(len(total_val_loss))
ax[0].plot(x, total_val_loss, label='total_val_loss')
ax[0].plot(x, val_loss_0, label='val_loss_0')
ax[0].plot(x, val_loss_1, label='val_loss_1')
ax[0].legend()
ax[1].plot(x, total_train_loss, label='total_train_loss')
ax[1].plot(x, train_loss_0, label='train_loss_0')
ax[1].plot(x, train_loss_1, label='train_loss_1')
ax[1].legend()

fig.tight_layout()
plt.savefig(f'/home/nwaftp23/scratch/train_val_loss.png')
plt.close()
