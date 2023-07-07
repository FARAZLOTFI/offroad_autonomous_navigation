import os
import torch.utils.data
from torch.utils.data.sampler import SequentialSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch import nn
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.engine import Engine
import numpy as np 

data_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/'
batch_size = 128

#model to load
model_path = '/usr/local/data/kvirji/offroad_autonomous_navigation/classifier/models/0/best.pt'

#data transformation
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#create dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory,x),
                                          data_transforms[x])
                  for x in ['test']}

#lengths of the dataset
N_test = len(image_datasets['test'])

#define sampler. for testing use a sequential sampler
test_sampler = SequentialSampler(image_datasets['test'])

#define data loader
test_loader = torch.utils.data.DataLoader(image_datasets['test'], sampler=test_sampler, batch_size=batch_size)

#load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

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

#load model from checkpoint 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

#define evaluators
def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

#precision
precision = Precision(average=False, device=device)
precision.attach(default_evaluator, "precision")

#recall
recall = Recall(average=False, device=device)
recall.attach(default_evaluator, "recall")

#f1
F1 = precision * recall * 2 / (precision + recall)
F1.attach(default_evaluator, "F1")


#eval
print("Starting evaluation on {}".format(device))
test_labels = None
test_outputs = None
with torch.no_grad(): #dont calculate gradients for test set
    model.eval()    #set model to eval mode 
    for j, (features, labels) in enumerate(tqdm(test_loader)): 
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)   #run model on inputs
                
        #store all outputs and labels to analyze precision, recall, and f1-score per epoch
        if j == 0:
            test_labels = labels
            test_outputs = outputs
        else:
            test_labels = torch.cat((test_labels, labels))
            test_outputs = torch.cat((test_outputs, outputs))
    
    #get test statistics 
    state = default_evaluator.run([[test_outputs, test_labels]])
    test_precision = state.metrics['precision']
    test_recall = state.metrics['recall']
    test_F1 = state.metrics['F1']
    np.set_printoptions(precision=4)
    print("------------------------------------- Results per class -------------------------------------")
    print("Classes: [Tree, Other Obstacle, Human, Waterhole, Mud, Jump, Grass, Smooth Road, Wet Leaves]")
    print("Precision: ", test_precision.detach().cpu().numpy())
    print("Recall: ", test_recall.detach().cpu().numpy())
    print("F1: ", test_F1.detach().cpu().numpy())