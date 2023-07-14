import torch
from torchvision import datasets, transforms, models
import os
import sys
import cv2
import time
import numpy as np
from models.nn_model import predictive_model_badgr
import random
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import time
from MHE_MPC.system_identification import euler_from_quaternion, GPS_deg2vel
import MHE_MPC.config as config
import matplotlib.pyplot as plt 
plt.ion()

image_size=(72,128)
classification_criterion = torch.nn.CrossEntropyLoss().cuda()
regression_criterion = torch.nn.MSELoss().cuda()

augment = transforms.Compose([
    transforms.Resize(size=image_size),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4, 0.3, 0.3, 0.3),
    transforms.GaussianBlur(kernel_size=(9,9)),
    ])

def load_topic_file(file_path, prev_gps_data):
    # the order of the data is: steering angle, throttle, w, x, y, z, Lon, lat,
    # then, timestamps for the image, depth, teensy, imu, and gps topics
    loaded_data = np.load(file_path)
    steering_angle, throttle, w, x, y, z, lon_GPS, lat_GPS = loaded_data[:8]
    roll, pitch, yaw = euler_from_quaternion(x, y, z, w)
    pitch = -pitch
    if prev_gps_data is None:
        # note, the following is provided in degree! also we don't have vel as
        # at this point there is only one GPS point available
        return np.array([lon_GPS, lat_GPS, pitch, steering_angle * (-0.6), throttle])
    else:  # we have to convert the degree to distance; the following gives us the distance in km
        vel, dbearing = GPS_deg2vel(prev_gps_data[0], lon_GPS, prev_gps_data[1], lat_GPS)
        return np.array([lon_GPS, lat_GPS, dbearing, steering_angle * (-0.6), throttle])


def input_preparation(images_list, images_path, topics_list, topics_path, classes_list, classes_path, planning_horizon, batchsize, debug_=False):
    image_batch = []
    actions_batch = []
    classes_batch = []
    orientations_batch = []
    candidates = np.random.randint(len(images_list), size=batchsize)
    # assuming that we got the current image and the next few sensor measurements
    # for each random candidate we read the image, then depending on the its index in the image list
    # we look for the correspondent topic and annotation in the topic and annotation list
    # as we need a set of these we go forward collecting topics and annotations for the future also ending up with
    # the required data for training the model
    for item in candidates:
        ##################################################################
        file_number = None
        flag_bag_changed = False
        image = cv2.imread(images_path+images_list[item])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Now prepare the ground truths
        set_of_actions = []
        set_of_orientations = []
        set_of_events = []
        gps_data = None # to be used for bearing estimation
        for i in range(planning_horizon + 1):
            # we have this assumption of correspendency between the images and the ground truth
            if not(item + i<len(topics_list)):
                flag_bag_changed = True
                break
            topic_file = topics_path + topics_list[item + i]
            ##### this is used to take care of the change in the rosbag #####
            if file_number is None:
                file_number = int(topics_list[item + i][-8:-4])
            else:
                if not(int(topics_list[item + i][-8:-4]) - file_number == 1):
                    print('Rosbag changed!!!')
                    flag_bag_changed = True
                    break  # if the rosbag is changed we need to skip the sample as it's not a proper one
                else:
                    file_number = int(topics_list[item + i][-8:-4])

            if gps_data is None:
                measurements = load_topic_file(topic_file, gps_data)
                # lon, lat
                gps_data = measurements[:2]
            else:
                measurements = load_topic_file(topic_file, gps_data)
                gps_data = measurements[:2]
                # note that we ignore the current orientation, we need the future ones
                set_of_orientations.append(measurements[2])
                # also the event/class of the future samples
                class_file = classes_path + classes_list[item + i]
                with open(class_file) as f:
                    image_class = int(float(f.read()))
                set_of_events.append(image_class)

            actions = measurements[-2:]

            if len(set_of_actions)<planning_horizon:
                # note that we ignore the horizon + 1 action
                set_of_actions.append(actions)

            if (debug_):
                plt.figure('augmented image')
                plt.imshow(image)

                print('image: ',images_list[item])
                print('topic: ', topics_list[item + i])
                print('class: ', classes_list[item + i])

                input('Press enter to continue..')

        if not flag_bag_changed:
            image_batch.append(image)
            actions_batch.append(np.array(set_of_actions))
            classes_batch.append(np.array(set_of_events))
            orientations_batch.append(np.array(set_of_orientations))

    image_batch = torch.from_numpy(np.array(image_batch)).float().cuda().permute(0,3,1,2)
    image_batch = augment(image_batch) #TODO to check!!

    actions_batch = torch.from_numpy(np.array(actions_batch)).float().cuda().permute(1,0,2)
    classes_batch = torch.from_numpy(np.array(classes_batch)).cuda().permute(1,0)
    orientations_batch = torch.from_numpy(np.array(orientations_batch)).float().cuda().permute(1,0)

    return [image_batch, actions_batch], [classes_batch, orientations_batch]  # input, output

def total_loss(planning_horizon, nn_out, true_out):
    train_loss1 = 0
    train_loss2 = 0
    for i in range(planning_horizon):
        train_loss1 += classification_criterion(nn_out[0][i], true_out[0][i])#
        train_loss2 += regression_criterion(nn_out[1][i], true_out[1][i])  # batch, ...

    train_loss = train_loss1 + train_loss2
    return train_loss,[train_loss1,train_loss2]

if __name__ == "__main__":

    load_from_checkpoint = False

    planning_horizon = 15
    num_of_events = 9 + 1
    action_dimension = 2
    model = predictive_model_badgr(planning_horizon, num_of_events, action_dimension)
    model.cuda()

    #summary(model, (4, 320, 240))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0008,weight_decay=0.000001)

    # isFile = os.path.isdir('trained_model')
    # if not(isFile):
    #     os.mkdir('trained_model')

    CHECKPOINT_PATH = '/home/barbados/checkpoint_weights_planner_July12/training_checkpoint'

    if(load_from_checkpoint):
        try:        
            checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            last_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Starting from the previous checkpoint. EPOCH: ',last_epoch)
            
        except:
            ##################### this i have to fix to be able to load weights from the trained classifier
            print('No checkpoint found!')
            last_epoch = 0
    else:
        last_epoch = 0
    # Writer will output to ./runs/ directory by default
    writer_train = SummaryWriter('/home/barbados/traj_planner_training_logs/runs/training')
    writer_val = SummaryWriter('/home/barbados/traj_planner_training_logs/runs/validation')

    other_train_writers = []
    other_val_writers = []

    monitored_terms_count = 2

    for ii in range(monitored_terms_count):
        other_train_writers.append(SummaryWriter('/home/barbados/traj_planner_training_logs/runs/'+'train_loss_term'+str(ii)))
        other_val_writers.append(SummaryWriter('/home/barbados/traj_planner_training_logs/runs/'+'val_loss_term'+str(ii)))

    path_to_images = config.path_to_dataset+'images/'
    path_to_topics = config.path_to_dataset+'topics/'
    path_to_annotations = config.path_to_dataset+'annotations/'

    images_list = os.listdir(path_to_images)
    topics_list = os.listdir(path_to_topics)
    annotations_list = os.listdir(path_to_annotations)

    images_list.sort()
    topics_list.sort()
    annotations_list.sort()

    train_images_list = images_list[:int(0.8*len(images_list))]
    val_images_list = images_list[int(0.8*len(images_list)):]

    train_topics_list = topics_list[:int(0.8 * len(topics_list))]
    val_topics_list = topics_list[int(0.8 * len(topics_list)):]

    train_classes_list = annotations_list[:int(0.8 * len(annotations_list))]
    val_classes_list = annotations_list[int(0.8 * len(annotations_list)):]

    BATCH_SIZE = 128
    epochs = range(last_epoch,300)
    train_iterations = int(len(train_images_list)/BATCH_SIZE)
    validation_iterations = int(len(val_images_list)/BATCH_SIZE)
    for param in model.parameters():
        param.requires_grad = True
    # clear the gradients
    for epoch in epochs:
        # Training part 
        model.train()

        for i in range(train_iterations):
            #images_list, images_path, topics_list, topics_path, classes_list, classes_path, planning_horizon, batchsize
            inputs, true_outputs = input_preparation(train_images_list, path_to_images, train_topics_list, path_to_topics,
                                               train_classes_list, path_to_annotations, planning_horizon, batchsize=BATCH_SIZE)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)

            # calculate loss
            train_loss, train_loss_terms = total_loss(planning_horizon, model_outputs, true_outputs)

            # credit assignment
            optimizer.zero_grad()
        
            train_loss.backward()
            # update model weights
            optimizer.step()

        # validation part
        model.eval()
        
        for i in range(validation_iterations):
            inputs, true_outputs = input_preparation(val_images_list, path_to_images, val_topics_list, path_to_topics,
                                               val_classes_list, path_to_annotations, planning_horizon, batchsize=BATCH_SIZE)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)
            # calculate loss
            val_loss, val_loss_terms = total_loss(planning_horizon, model_outputs, true_outputs)
            #print("Processing the training data: ",100*zz/len(val_list),' validation loss: ',val_loss, end='', flush=True)


        writer_train.add_scalar('Total Loss', train_loss, epoch)
        writer_val.add_scalar('Total Loss', val_loss, epoch)
        for ii in range(monitored_terms_count):
            other_train_writers[ii].add_scalar('Loss Term'+str(ii),train_loss_terms[ii], epoch)
            other_val_writers[ii].add_scalar('Loss Term'+str(ii),val_loss_terms[ii], epoch)

        if epoch%1 == 0: # save the model every 2 epochs
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),} , CHECKPOINT_PATH) # model.block_junction.

        print(epoch," ",train_loss,val_loss)


