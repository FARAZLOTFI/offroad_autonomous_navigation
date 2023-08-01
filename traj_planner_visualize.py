import torch
from torchvision import transforms
import os
import sys
from models.nn_model import PredictiveModelBadgr, LSTMSeqModel
from traj_planner_helpers import load_data_lists, input_preparation
import MHE_MPC.config as config
import matplotlib.pyplot as plt 
from metrics import Metrics
import cv2 

plt.ion()

image_size=(72,128)

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.ToTensor()
    ])

if __name__ == "__main__":

    planning_horizon = 20
    num_event_types = 9 + 1  # one for regression
    n_seq_model_layers = 4
    seq_elem_dim = 16
    action_dimension = 2
    seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
    model = PredictiveModelBadgr(planning_horizon, num_event_types,
                                            action_dimension, seq_encoder, n_seq_model_layers)
    model.cuda()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CHECKPOINT_PATH = config.model_checkpoint+'training_checkpoint_wreg_20'
   
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
   
    start_sample = 112
    # start_sample = images_list.index('image_2023-06-14-10-40-39_0020.jpg')
    
    metrics = Metrics(planning_horizon=planning_horizon, device=device)

    with torch.no_grad(): #dont calculate gradients for test set
        
        #set model to eval mode
        model.eval()

        metrics.reset()

        inputs, true_outputs = input_preparation(images_list, path_to_images, topics_list, path_to_topics,
                                            annotations_list, path_to_annotations, planning_horizon, batchsize=1, augment=augment, randomize=False, start_sample=start_sample)
        # compute the model output
        model_outputs = model.training_phase_output(inputs)
        
        metrics.update(model_outputs, true_outputs)

        true_events = true_outputs[0].squeeze().detach().cpu().numpy() 
        pred_events = torch.argmax(model_outputs[0].squeeze(), dim=1).detach().cpu().numpy()

        true_orient = true_outputs[1].squeeze().detach().cpu().numpy()
        pred_orient = model_outputs[1].squeeze().detach().cpu().numpy()

        print()

        print('ANALYZING IMAGE: ', images_list[start_sample])
        print('TRUE EVENTS: ', true_events)
        print('PREDICTED EVENTS: ', pred_events)
        print('TRUE ORIENTATIONS: ', true_orient)
        print('PREDICTED ORIENTATIONS: ', pred_orient)

        print()
        
        metrics.compute()

        im = inputs[0].squeeze().permute(1,2,0).detach().cpu().numpy()
        cv2.imshow('Starting Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        