import os
import torch.utils.data
from torch.utils.data.sampler import SequentialSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch import nn
from ignite.metrics import Precision, Recall, ConfusionMatrix, Accuracy
import numpy as np 
from matplotlib import pyplot as plt 
import cv2 


use_augmented = False
data_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/dataset/'
batch_size = 128
label_names = ['Tree', 'Other Obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable Grass', 'Smooth Road', 'Wet Leaves']
save_misclassified = False

#model to load
model_dir = '/usr/local/data/kvirji/offroad_autonomous_navigation/classifier/models/0/'
model_fname = 'best.pt'

#data transformation
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

revert_normalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

#create dataset
if use_augmented:
    test_set = 'test_augmented'
else:
    test_set = 'test'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory,x),
                                          data_transforms[x])
                  for x in [test_set]}

#length of the dataset and number of classes
N_test = len(image_datasets[test_set])
N_classes = len(image_datasets[test_set].classes)

#define sampler. for testing use a sequential sampler
test_sampler = SequentialSampler(image_datasets[test_set])

#define data loader
test_loader = torch.utils.data.DataLoader(image_datasets[test_set], sampler=test_sampler, batch_size=batch_size)

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
checkpoint = torch.load(os.path.join(model_dir, model_fname), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

#metrics
precision = Precision(average=False, device=device)
recall = Recall(average=False, device=device)
F1 = precision * recall * 2 / (precision + recall)
cm = ConfusionMatrix(num_classes=N_classes)
accuracy = Accuracy(device=device)

#eval
print("Starting evaluation on {}".format(device))
with torch.no_grad(): #dont calculate gradients for test set
    model.eval()    #set model to eval mode 
    for j, (features, labels) in enumerate(tqdm(test_loader)): 
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)   #run model on inputs
        
        #update metrics
        precision.update((outputs,labels))
        recall.update((outputs,labels))
        F1.update((outputs,labels))
        cm.update((outputs,labels))
        accuracy.update((outputs,labels))

        #save misclassified examples
        if save_misclassified:
            _, preds = torch.max(outputs,1)
            idxs_mask = ((preds == labels) == False).nonzero()
            for i in idxs_mask:
                    cv_image = revert_normalize(features[i]).movedim(1,-1).detach().cpu().numpy().squeeze().copy()
                    cv2.putText(cv_image, "Pred: {}, Actual: {}".format(label_names[preds[i]], label_names[labels[i]]), (10, 200), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(model_dir, 'misclassified_{}/misclassified_{}_{}.png'.format(test_set, i.item(),j)), 255*cv_image)
                
    #compute test statistics 
    precision = precision.compute().detach().cpu().numpy()
    recall = recall.compute().detach().cpu().numpy()
    F1 = F1.compute().detach().cpu().numpy()
    cm = cm.compute().detach().cpu().numpy()
    accuracy = accuracy.compute()

    np.set_printoptions(precision=4)
    print("------------------------------------- Results per class on {} set-------------------------------------".format(test_set))
    print("Classes: ", label_names)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)
    print("Accuracy: {:.4f}".format(accuracy))

    fig, ax = plt.subplots(figsize=(16,16))
    ax.matshow(cm, cmap=plt.cm.Greens)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(N_classes), label_names)
    ax.set_yticks(np.arange(N_classes), label_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='large')
    
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix_{}.png'.format(test_set)))
    print('Saved Confusion Matrix')

