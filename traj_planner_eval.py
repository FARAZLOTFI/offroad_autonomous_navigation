import torch
from torchvision import transforms
import os
import sys
from models.nn_model import PredictiveModelBadgr, LSTMSeqModel
from tqdm import tqdm
from traj_planner_helpers import load_data_lists, input_preparation
import MHE_MPC.config as config
import matplotlib.pyplot as plt 
from metrics import Metrics
plt.ion()

image_size=(72,128)

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0),
    # transforms.GaussianBlur(kernel_size=(9,9), sigma=(1e-10,2)), 
    transforms.ToTensor()
    ])

if __name__ == "__main__":

    planning_horizon = 5
    num_event_types = 9 + 1  # one for regression
    n_seq_model_layers = 4
    seq_elem_dim = 16
    action_dimension = 2
    seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
    model = PredictiveModelBadgr(planning_horizon, num_event_types,
                                            action_dimension, seq_encoder, n_seq_model_layers)
    model.cuda()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CHECKPOINT_PATH = config.model_checkpoint+'training_checkpoint_617'
   
    try:        
        checkpoint = torch.load(CHECKPOINT_PATH,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        last_epoch = checkpoint['epoch']
        print('Starting from the previous checkpoint. EPOCH: ',last_epoch)
    except:
        print('No checkpoint found!')
        sys.exit()

    path_to_images = config.path_to_dataset+'images/'
    path_to_topics = config.path_to_dataset+'topics/'
    path_to_annotations = config.path_to_dataset+'annotations/'

    images_list = os.listdir(path_to_images)
    topics_list = os.listdir(path_to_topics)
    annotations_list = os.listdir(path_to_annotations)

    images_list.sort()
    topics_list.sort()
    annotations_list.sort()
   
    training_samples, validation_samples = load_data_lists(len(images_list))

    train_images_list = [images_list[i] for i in training_samples]
    val_images_list = [images_list[i] for i in validation_samples]

    BATCH_SIZE = 64
    train_iterations = int(len(train_images_list)/BATCH_SIZE)
    validation_iterations = int(len(val_images_list)/BATCH_SIZE)

    metrics = Metrics(planning_horizon=planning_horizon, device=device)
    
    with torch.no_grad(): #dont calculate gradients for test set
        
        # training set
        model.eval()
        
        #reset metrics
        metrics.reset()

        for i in tqdm(range(train_iterations)):
            #images_list, images_path, topics_list, topics_path, classes_list, classes_path, planning_horizon, batchsize
            inputs, true_outputs = input_preparation(train_images_list, path_to_images, topics_list, path_to_topics,
                                                annotations_list, path_to_annotations, planning_horizon, batchsize=BATCH_SIZE, augment=augment)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)

            #update metrics
            metrics.update(model_outputs[0], true_outputs[0])

        metrics.compute(filename='metrics_train.png')



        # validation set    
        metrics.reset()

        for i in tqdm(range(validation_iterations)):
            inputs, true_outputs = input_preparation(val_images_list, path_to_images, topics_list, path_to_topics,
                                                annotations_list, path_to_annotations, planning_horizon, batchsize=BATCH_SIZE, augment=augment)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)
           
            metrics.update(model_outputs[0], true_outputs[0])

        metrics.compute(filename='metrics_val.png')
       
      


