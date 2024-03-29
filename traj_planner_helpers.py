import os
import torch
import cv2
import numpy as np
from MHE_MPC.system_identification import euler_from_quaternion, GPS_deg2vel
import MHE_MPC.config as config
import matplotlib.pyplot as plt 

LOG_SIG_MAX = 0.5
LOG_SIG_MIN = -0.5


def load_data_lists(num_images, path):
    # validation samples
    try:
        validation_samples = []
        with open(os.path.join(path, "validation_samples.txt"), 'r') as f:
            lines_ = f.readlines()
            for line_ in lines_:
                validation_samples.append(int(line_))

        training_samples = []
        with open(os.path.join(path, "training_samples.txt"), 'r') as f:
            lines_ = f.readlines()
            for line_ in lines_:
                training_samples.append(int(line_))
        print('Samples list found!')
    except:
        print('There is no samples list found!')
        validation_samples = np.random.randint(num_images, size=int(0.2 * num_images))
        with open(os.path.join(path,  "validation_samples.txt"), 'w') as f:
            for item in validation_samples:
                f.write(str(item) + '\n')

        # Training samples
        training_samples = []
        for i in range(num_images):
            flag_item = True
            for item in validation_samples:
                if i == item:
                    flag_item = False
                    break
            if flag_item:
                training_samples.append(i)

        with open(os.path.join(path, "training_samples.txt"), 'w') as f:
            for item in training_samples:
                f.write(str(item) + '\n')
    return training_samples, validation_samples


def load_topic_file(file_path, prev_gps_data, preloaded_data = {}):
    # the order of the data is: steering angle, throttle, w, x, y, z, Lon, lat,
    # then, timestamps for the image, depth, teensy, imu, and gps topics
    #import pdb; pdb.set_trace()
    if not preloaded_data:
        loaded_data = np.load(file_path)
    else:
        loaded_data = preloaded_data[os.path.basename(file_path)]
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

def input_preparation(images_list, images_path, topics_list, topics_path, classes_list, classes_path,
        planning_horizon, batchsize, augment, randomize=True, start_sample = 0, 
        debug_=False, all_topics={}):
    image_batch = []
    actions_batch = []
    classes_batch = []
    orientations_batch = []
    if randomize:
        candidates = np.random.randint(len(images_list), size=batchsize)
    else:
        candidates = np.arange(start_sample, min(start_sample + batchsize, len(images_list) - planning_horizon))

    # assuming that we got the current image and the next few sensor measurements
    # for each random candidate we read the image, then depending on the its index in the image list
    # we look for the correspondent topic and annotation in the topic and annotation list
    # as we need a set of these we go forward collecting topics and annotations for the future also ending up with
    # the required data for training the model
    for candidate in candidates:
        ##################################################################
        file_number = None
        flag_bag_changed = False
        image = cv2.imread(images_path+images_list[candidate])
        item = topics_list.index('topics' + images_list[candidate][5:-4] + '.npy')
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        skip_step = np.random.randint(1,4)
        # Now prepare the ground truths
        set_of_actions = []
        set_of_orientations = []
        set_of_events = []
        gps_data = None # to be used for bearing estimation
        for i in range(planning_horizon + 1):
            # we have this assumption of correspendency between the images and the ground truth
            if not(item + i*skip_step<len(topics_list)):
                flag_bag_changed = True
                break
            topic_file = topics_path + topics_list[item + i*skip_step]
            ##### this is used to take care of the change in the rosbag #####
            if file_number is None:
                file_number = int(topics_list[item + i*skip_step][-8:-4])
            else:
                if not(int(topics_list[item + i*skip_step][-8:-4]) - file_number == 1*skip_step):
                    #print('Rosbag changed!!!')
                    flag_bag_changed = True
                    break  # if the rosbag is changed we need to skip the sample as it's not a proper one
                else:
                    file_number = int(topics_list[item + i*skip_step][-8:-4])

            if gps_data is None:
                measurements = load_topic_file(topic_file, gps_data, preloaded_data=all_topics)
                # lon, lat
                gps_data = measurements[:2]
            else:
                measurements = load_topic_file(topic_file, gps_data, preloaded_data=all_topics)
                gps_data = measurements[:2]
                # note that we ignore the current orientation, we need the future ones
                set_of_orientations.append(measurements[2]/np.pi)
                # also the event/class of the future samples
                class_file = classes_path + classes_list[item + i*skip_step]
                with open(class_file) as f:
                    image_class = int(float(f.read()))
                set_of_events.append(image_class)

            actions = measurements[-2:]
            ## changing the throttle according to the skip_step
            actions[-1] = actions[-1]*(skip_step/3)
            if len(set_of_actions)<planning_horizon:
                # note that we ignore the horizon + 1 action
                set_of_actions.append(actions)
            #debug_ = True
            if (debug_ ):
                plt.figure('augmented image')
                plt.imshow(image)

                print('image: ',images_list[candidate])
                print('topic: ', topics_list[item + i*skip_step])
                print('class: ', classes_list[item + i*skip_step])
                print('skip_step: ', skip_step)
                input('Press enter to continue..')

        if not flag_bag_changed:
            image_batch.append(augment(image))
            actions_batch.append(np.array(set_of_actions))
            classes_batch.append(np.array(set_of_events))
            orientations_batch.append(np.array(set_of_orientations))

    # image_batch = torch.from_numpy(np.array(image_batch)/255).float().cuda().permute(0,3,1,2)
    image_batch = torch.stack(image_batch).cuda() #TODO to check!!

    actions_batch = torch.from_numpy(np.array(actions_batch)).float().cuda().permute(1,0,2)
    classes_batch = torch.from_numpy(np.array(classes_batch)).cuda().permute(1,0)
    orientations_batch = torch.from_numpy(np.array(orientations_batch)).float().cuda().permute(1,0)

    return [image_batch, actions_batch], [classes_batch, orientations_batch]  # input, output


def total_loss(planning_horizon, classification_criterion, regression_criterion, nn_out, true_out, gaussian_crit=False):
    train_loss1 = 0
    train_loss2 = 0
    for i in range(planning_horizon):
        train_loss1 += classification_criterion(nn_out[0][i], true_out[0][i])#
        if gaussian_crit:
            log_sig = torch.exp(torch.clamp(nn_out[1][i][:,1], min=LOG_SIG_MIN, max=LOG_SIG_MAX))
            train_loss2 += regression_criterion(nn_out[1][i][:,0], true_out[1][i], torch.exp(log_sig))  # batch, ...
        else:
            train_loss2 += regression_criterion(nn_out[1][i], true_out[1][i])  # batch, ...

    train_loss = train_loss1 + train_loss2
    return train_loss,[train_loss1,train_loss2]


