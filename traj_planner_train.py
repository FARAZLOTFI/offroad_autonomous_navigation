import pickle
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import os
from models.nn_model import PredictiveModelBadgr, LSTMSeqModel, TransformerSeqModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import MHE_MPC.config as config
import matplotlib.pyplot as plt 
from metrics import Metrics
from torchsummary import summary
from traj_planner_helpers import load_data_lists, input_preparation, total_loss
import datetime
plt.ion()

image_size=(72,128)
classification_criterion = torch.nn.CrossEntropyLoss().cuda()
regression_criterion = torch.nn.MSELoss().cuda()

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0),
    # transforms.GaussianBlur(kernel_size=(9,9), sigma=(1e-10,2)), 
    transforms.ToTensor()
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=config.path_to_dataset,
                        help='path to dataset directory')
    parser.add_argument('--ensemble_size', default=1, type=int,
                        help='ensemble size for uncertainty estimation')
    parser.add_argument('--ensemble_type', default="fixed_masks", type=str,
                        help='type of ensemble to make')
    parser.add_argument('--seq_encoder', default="LSTM", type=str,
                        help='type of sequence encoder')
    parser.add_argument('--checkpoint_path', default="", type=str,
                        help='path to saved model')
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get the current date
    current_date = datetime.datetime.now()

    # format the date as dd-mm-yy
    formatted_date = current_date.strftime('%d_%m_%y_%H_%M_%S')
    config.training_logfiles = os.path.join(config.training_logfiles, f'offroad_trainfiles_{args.seq_encoder}_{formatted_date}')
    load_from_checkpoint = False
    if args.checkpoint_path:
        load_from_checkpoint = True
        config.training_logfiles = os.path.dirname(os.path.dirname(args.checkpoint_path)) 
    planning_horizon = 40 
    num_event_types = 9 + 1  # one for regression
    n_seq_model_layers = 4
    seq_elem_dim = 16
    action_dimension = 2
    if args.seq_encoder == 'LSTM':
        seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
    else:
        seq_encoder = TransformerSeqModel(n_seq_model_layers, seq_elem_dim)

    model = PredictiveModelBadgr(planning_horizon, num_event_types,
                                    action_dimension, seq_encoder, n_seq_model_layers,
                                    device = device, ensemble_size = args.ensemble_size, 
                                    ensemble_type = args.ensemble_type)
    ensemble_bool = False
    if args.ensemble_size > 1:
        regression_criterion = torch.nn.GaussianNLLLoss().cuda()
        ensemble_bool = True
    model.cuda()

    #summary(model, (4, 320, 240))
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0008,weight_decay=0.000001)
    # isFile = os.path.isdir('trained_model')
    # if not(isFile):
    #     os.mkdir('trained_model')

    model_dir = os.path.join(config.training_logfiles, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    CHECKPOINT_PATH = os.path.join(model_dir, 'training_checkpoint.pt')

    if (load_from_checkpoint):
        try:        
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
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
    writer_train = SummaryWriter(os.path.join(config.training_logfiles, 
        f'traj_planner_training_logs/runs/training'))
    writer_val = SummaryWriter(os.path.join(config.training_logfiles, 
        f'traj_planner_training_logs/runs/validation'))

    other_train_writers = []
    other_val_writers = []

    monitored_terms_count = 2

    for ii in range(monitored_terms_count):
        other_train_writers.append(SummaryWriter(os.path.join(config.training_logfiles, 
            f'traj_planner_training_logs/runs/'+'train_loss_term'+str(ii))))
        other_val_writers.append(SummaryWriter(os.path.join(config.training_logfiles, 
            f'traj_planner_training_logs/runs/'+'val_loss_term'+str(ii))))

    path_to_images = os.path.join(config.path_to_dataset, 'images/')
    path_to_topics = os.path.join(config.path_to_dataset, 'topics/')
    path_to_annotations = os.path.join(config.path_to_dataset, 'annotations/')

    images_list = os.listdir(path_to_images)
    topics_list = os.listdir(path_to_topics)
    annotations_list = os.listdir(path_to_annotations)
    #################### NOTE WE NEED TO SORT THESE LISTS TO FEED THE NN WITH THE RIGHT DATA#########
    images_list.sort()
    topics_list.sort()
    annotations_list.sort()
    ################################################################################################
    # The following was not good as we tend to take the first recorded bags for training, whereas the last ones for validation
    # train_images_list = images_list[:int(0.8*len(images_list))]
    # val_images_list = images_list[int(0.8*len(images_list)):]
    #
    # train_topics_list = topics_list[:int(0.8 * len(topics_list))]
    # val_topics_list = topics_list[int(0.8 * len(topics_list)):]
    #
    # train_classes_list = annotations_list[:int(0.8 * len(annotations_list))]
    # val_classes_list = annotations_list[int(0.8 * len(annotations_list)):]

    training_samples, validation_samples = load_data_lists(len(images_list), path=config.working_dir)

    train_images_list = [images_list[i] for i in training_samples]
    val_images_list = [images_list[i] for i in validation_samples]

    BATCH_SIZE = 64
    epochs = range(last_epoch,1200)
    train_iterations = int(len(train_images_list)/BATCH_SIZE)
    validation_iterations = int(len(val_images_list)/BATCH_SIZE)

    metrics = Metrics(planning_horizon=planning_horizon, device=device)
    try:
        with open('all_topics.pkl', 'rb') as f:
            all_topics = pickle.load(f)
            print('List of topics loaded!')
    except:
        all_topics = {}     
    for param in model.parameters():
        param.requires_grad = True
    # clear the gradients
    for epoch in epochs:
        # Training part 
        model.train()

        
        for i in tqdm(range(train_iterations)):
        #for i in tqdm(range(4)):
            #images_list, images_path, topics_list, topics_path, classes_list, classes_path, planning_horizon, batchsize
            inputs, true_outputs = input_preparation(train_images_list, path_to_images, 
                topics_list, path_to_topics, annotations_list, path_to_annotations, 
                planning_horizon, batchsize=BATCH_SIZE, augment=augment, all_topics = all_topics)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)

            # calculate loss
            train_loss, train_loss_terms = total_loss(planning_horizon, classification_criterion, 
                regression_criterion, model_outputs, true_outputs, gaussian_crit=ensemble_bool)

            # credit assignment
            optimizer.zero_grad()
        
            train_loss.backward()
            # update model weights
            optimizer.step()

        # validation part
        model.eval()
        
        #reset metrics
        metrics.reset()
        
        epi_unc_rg = []
        epi_unc_clf = []
        for i in tqdm(range(validation_iterations)):
        #for i in tqdm(range(4)):
            inputs, true_outputs = input_preparation(val_images_list, path_to_images, 
                topics_list, path_to_topics, annotations_list, path_to_annotations, 
                planning_horizon, batchsize=BATCH_SIZE, augment=augment, all_topics = all_topics)
            # compute the model output
            model_outputs = model.training_phase_output(inputs)
            # calculate loss
            val_loss, val_loss_terms = total_loss(planning_horizon, classification_criterion, 
                regression_criterion, model_outputs, true_outputs, gaussian_crit=ensemble_bool)
            #print("Processing the training data: ",100*zz/len(val_list),' validation loss: ',val_loss, end='', flush=True)

            #update metrics with batch data
            metrics.update(model_outputs, true_outputs, gauss_out = ensemble_bool)
            if ensemble_bool:
                pred_classification, pred_regression, epi_unc_classification, epi_unc_regressions = metrics.calc_unc(model, inputs)
                epi_unc_clf.append(epi_unc_classification)
                epi_unc_rg.append(epi_unc_regressions)
        
        #print metrics but dont save
        metrics.compute()
        if ensemble_bool:
            epi_unc_clf = torch.hstack(epi_unc_clf)
            KL = [i['KL'] for i in epi_unc_rg]
            BHATT = [i['Bhatt'] for i in epi_unc_rg]
            epi_unc_rg = {'KL':torch.hstack(KL), 'Bhatt': BHATT}
            metrics.plot_unc(epi_unc_clf, epi_unc_regressions, args.ensemble_type, args.ensemble_size, 
                config.training_logfiles, epoch, args.seq_encoder)

       
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

        print(epoch,f" train_loss:{train_loss.item()}, val_loss:{val_loss.item()}")


