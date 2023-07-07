import os
import torch.utils.data
from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch import nn, optim
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.engine import Engine
import numpy as np 
from matplotlib import pyplot as plt

data_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/'
model_save_path = '/usr/local/data/kvirji/offroad_autonomous_navigation/classifier/models/0/'
batch_size = 128
epochs = 100

#to load from checkpoint
load_checkpoint = False
checkpoint_path = '/usr/local/data/kvirji/offroad_autonomous_navigation/classifier/models/0/last.pt'

#set to true to analyze sample distribution per batch and view image augmentations
analyze = False

#data transformations
data_transforms = {
    'train': transforms.Compose([   #only augment training data
        transforms.Resize(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.2, 0.2, 0.1),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory,x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

#lengths of the datasets
N_train = len(image_datasets['train'])
N_val = len(image_datasets['val'])

#define samplers. for training use a weighted sampler. for validation use a sequential sampler
class_counts = torch.unique(torch.tensor(image_datasets['train'].targets), return_counts=True)[1].detach().cpu().numpy()
class_weights = [1/c for c in class_counts]     #flexible to choose class weights
sample_weights = [class_weights[t] for t in image_datasets['train'].targets]
train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=N_train, replacement=True)
val_sampler = SequentialSampler(image_datasets['val'])

#define data loaders
train_loader = torch.utils.data.DataLoader(image_datasets['train'], sampler=train_sampler, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(image_datasets['val'], sampler=val_sampler, batch_size=batch_size)

#load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

# freeze low layer parameters 
for param in model.parameters():
    param.requires_grad = False

#last layer - flexible to change this 
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), 
    nn.Dropout(0.4), 
    nn.Linear(256,9), # 9 = number of classes
    nn.LogSoftmax(dim=1)
)

#send model to device
model.to(device)

#define loss function and optimizer. flexible to change
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

best_F1 = 0.0
last_epoch = 0
history = []

#load model from checkpoint 
if load_checkpoint: 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    last_epoch = checkpoint['epoch']
    last_train_precision = checkpoint['train_precision']
    last_train_recall = checkpoint['train_recall'] 
    last_train_F1 = checkpoint['train_F1'] 
    last_train_loss = checkpoint['train_loss'] 
    last_valid_precision = checkpoint['valid_precision']
    last_valid_recall = checkpoint['valid_recall'] 
    last_valid_F1 = checkpoint['valid_F1'] 
    last_valid_loss = checkpoint['valid_loss'] 
    history = checkpoint['history']
    print("Loaded pretrained model")
    print("Last Training Stats: Epoch {}, Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(last_epoch, last_train_loss, last_train_precision, last_train_recall, last_train_F1))
    print("Last Valid Stats: Epoch {}, Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(last_epoch, last_valid_loss, last_valid_precision, last_valid_recall, last_valid_F1))
    best_F1 = last_valid_F1

#analyze training batches
if analyze:
    import cv2
    import time
    print('Analyzing first 10 batches')
    label_counts_per_batch = []
    for step, (features,labels) in enumerate(train_loader):
        if step == 10: 
            for feature in features: #display all images in batch
                cv2.imshow('Image', feature.movedim(0,-1).detach().cpu().numpy())
                cv2.waitKey(250)
                time.sleep(0.25)
            break
        label_counts_per_batch.append(list(torch.unique(labels, return_counts=True)[1].detach().cpu().numpy())) #save label counts per batch
        
    label_counts_per_batch = np.asarray(label_counts_per_batch)
    plt.figure(figsize=(15, 6))
    cls = ['Tree', 'Other Obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable Grass', 'Smooth Road', 'Wet Leaves']
    plt.bar(cls, np.mean(label_counts_per_batch, axis=0), yerr=np.std(label_counts_per_batch, axis=0) )
    plt.title("Average distribution of samples per batch")
    plt.ylabel("Number of samples")
    plt.xlabel("Count")
    plt.show()

#define evaluators
def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

#precision
macro_precision = Precision(average=True, device=device)
macro_precision.attach(default_evaluator, "macro_precision")

#recall
macro_recall = Recall(average=True, device=device)
macro_recall.attach(default_evaluator, "macro_recall")

#f1
precision = Precision(average=False, device=device)
recall = Recall(average=False, device=device)
macro_F1 = (precision * recall * 2 / (precision + recall)).mean()
macro_F1.attach(default_evaluator, "macro_F1")

#training loop
print("Starting training for {} epochs on {}".format(epochs, device))
for epoch in range(last_epoch + 1, last_epoch + epochs + 1):
    print("Epoch: {}/{}".format(epoch, last_epoch + epochs))

    model.train()   #set model to training mode
    train_loss = 0.0
    train_labels = None
    train_outputs = None
    for i, (features, labels) in enumerate(tqdm(train_loader)):
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()   #clear gradients
        outputs = model(features)   #run model on inputs
        loss = loss_fn(outputs, labels)     #calculate loss between predictions and ground truth
        loss.backward()     #backpropagate loss
        optimizer.step()    #update weights
        train_loss += loss.item() * features.size(0)    #multiply loss by batch size and add it to running count of the training loss 
        
        #store all outputs and labels to analyze precision, recall, and f1-score per epoch
        if i == 0:
            train_labels = labels
            train_outputs = outputs
        else:
            train_labels = torch.cat((train_labels, labels))
            train_outputs = torch.cat((train_outputs, outputs))
    
    #get training statistics 
    state = default_evaluator.run([[train_outputs, train_labels]])
    train_precision = state.metrics['macro_precision']
    train_recall = state.metrics['macro_recall']
    train_F1 = state.metrics['macro_F1']
    avg_train_loss = train_loss/N_train #average loss per sample in training set
    print("Training Stats: Epoch {}, Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(epoch, avg_train_loss, train_precision, train_recall, train_F1))
    
    #evaluate model on validation set
    valid_loss = 0.0
    valid_labels = None
    valid_outputs = None
    with torch.no_grad(): #dont calculate gradients for validation set
        model.eval()    #set model to eval mode 
        for j, (features, labels) in enumerate(tqdm(val_loader)): 
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)   #run model on inputs
            loss = loss_fn(outputs,labels)  #calculate loss between predictions and ground truth
            valid_loss += loss.item() * features.size(0)    #multiply loss by batch size and add it to the running count of the validation loss
            
            #store all outputs and labels to analyze precision, recall, and f1-score per epoch
            if j == 0:
                valid_labels = labels
                valid_outputs = outputs
            else:
                valid_labels = torch.cat((valid_labels, labels))
                valid_outputs = torch.cat((valid_outputs, outputs))
        
        #get valid statistics 
        state = default_evaluator.run([[valid_outputs, valid_labels]])
        valid_precision = state.metrics['macro_precision']
        valid_recall = state.metrics['macro_recall']
        valid_F1 = state.metrics['macro_F1']
        avg_valid_loss = valid_loss/N_val #average loss per sample in validation set
        print("Valid Stats: Epoch {}, Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(epoch, avg_valid_loss, valid_precision, valid_recall, valid_F1))

    #append to history
    history.append([epoch, avg_train_loss, train_precision, train_recall, train_F1, avg_valid_loss, valid_precision, valid_recall, valid_F1])
    
    #save best model based on f1 score
    if valid_F1 >= best_F1: 
        print('Updating best model')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'train_precision': train_precision, 
            'train_recall': train_recall, 
            'train_F1': train_F1, 
            'train_loss': avg_train_loss,
            'valid_precision': valid_precision, 
            'valid_recall': valid_recall, 
            'valid_F1': valid_F1, 
            'valid_loss': avg_valid_loss,
            'history': history
        }, os.path.join(model_save_path, 'best.pt'))
        best_F1 = valid_F1

    #save most recent model for continued training
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'train_precision': train_precision, 
        'train_recall': train_recall, 
        'train_F1': train_F1, 
        'train_loss': avg_train_loss,
        'valid_precision': valid_precision, 
        'valid_recall': valid_recall, 
        'valid_F1': valid_F1, 
        'valid_loss': avg_valid_loss,
        'history': history
    }, os.path.join(model_save_path, 'last.pt'))

history = np.asarray(history)
x = history[:,0].astype(int)
plt.plot(x, history[:,1], label='Training')
plt.plot(x, history[:,5], label='Validation')
plt.title('Learning Curves')
plt.ylabel('Negative log likelihood loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(model_save_path + 'learning_curves.png')
plt.clf()

plt.plot(x, history[:,4], label='Training')
plt.plot(x, history[:,8], label='Validation')
plt.title('Macro F1-Scores')
plt.ylabel('F1-score')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(model_save_path + 'f1_scores.png')
plt.clf()
