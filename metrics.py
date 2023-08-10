from ignite.metrics import Precision, Recall, Accuracy, MeanSquaredError, ConfusionMatrix
from matplotlib import pyplot as plt
import numpy as np 

class Metrics: 

    #initialize metrics
    def __init__(self, planning_horizon, device):
        self.labels = ['Tree', 'Other Obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable Grass', 'Smooth Road', 'Wet Leaves']
        self.planning_horizon = planning_horizon
        self.device = device
        self.recall = []
        self.precision = []
        self.f1 = []
        self.accuracy = []
        self.mse = []
        self.cm = []
        for _ in range(planning_horizon):
            self.recall.append(Recall(average=True, device=self.device))
            self.precision.append(Precision(average=True, device=self.device))
            p = Precision(device=self.device, average=False)
            r = Recall(device=self.device, average=False)
            f1 = (p * r * 2 / (p + r))
            self.f1.append(f1)
            self.accuracy.append(Accuracy(device=self.device))
            self.mse.append(MeanSquaredError(device=self.device))
            self.cm.append(ConfusionMatrix(num_classes=len(self.labels), device=self.device))

    #reset all metrics
    def reset(self): 
        for i in range(self.planning_horizon): 
            #classification metrics
            self.recall[i].reset()
            self.precision[i].reset()
            self.f1[i].reset()
            self.accuracy[i].reset()
            self.cm[i].reset()
            
            #regression metrics
            self.mse[i].reset()
        return
    
    #update all metrics with batch data
    def update(self, outputs, labels):
        for i in range(self.planning_horizon): 
            #classification metrics
            self.precision[i].update((outputs[0][i],labels[0][i]))
            self.recall[i].update((outputs[0][i],labels[0][i]))
            self.accuracy[i].update((outputs[0][i],labels[0][i]))
            self.f1[i].update((outputs[0][i],labels[0][i]))
            self.cm[i].update((outputs[0][i],labels[0][i]))

            #regression metrics
            self.mse[i].update((outputs[1][i], labels[1][i]))

        return
    
    #compute metrics with data passed in through update
    def compute(self, regression_filename=None, classification_filename=None, classification_title='Classification Metrics', regression_title='Regression Metrics'):
        p = []
        r = []
        f1 = []
        acc = []
        x = []
        mse = []
        cm = []
        for i in range(self.planning_horizon):
            x.append(i + 1)
            p.append(self.precision[i].compute())
            r.append(self.recall[i].compute())
            acc.append(self.accuracy[i].compute())
            mse.append(self.mse[i].compute())
            cm.append(self.cm[i].compute().detach().cpu().numpy())
            f1.append(np.mean(np.nan_to_num(self.f1[i].compute().detach().numpy())))   

        print("---METRICS---\nPrecision: {}\nRecall: {}\nF1: {}\nAccuracy: {}\nMSE: {}\n".format(p, r, f1, acc, mse))

        if classification_filename:
            plt.plot(x, p, label='Precision', marker='o')
            plt.plot(x, r, label='Recall',marker='o')
            plt.plot(x, acc, label='Accuracy',marker='o')
            plt.plot(x, f1, label="F1", marker='o')
            plt.xlabel('Planning Horizon')
            plt.ylabel('Score')
            plt.title(classification_title)
            plt.legend()
            plt.savefig(classification_filename)
            plt.cla()

        if regression_filename:
            plt.plot(x, mse, marker='o')
            plt.xlabel('Planning Horizon')
            plt.ylabel('MSE')
            plt.title(regression_title)
            plt.savefig(regression_filename)
            plt.cla()

        # for ind,matrix in enumerate(cm): 
        #     fig, ax = plt.subplots(figsize=(16,16))
        #     ax.matshow(matrix, cmap=plt.cm.Greens)
        #     ax.xaxis.set_ticks_position('bottom')
        #     ax.set_xticks(np.arange(len(self.labels)), self.labels)
        #     ax.set_yticks(np.arange(len(self.labels)), self.labels)

        #     for i in range(matrix.shape[0]):
        #         for j in range(matrix.shape[1]):
        #             ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='large')
            
        #     plt.xlabel('Predictions')
        #     plt.ylabel('Ground Truth')
        #     plt.title('Confusion Matrix')
        #     plt.savefig('confusion_matrix_{}.png'.format(ind))
        # print('Saved Confusion Matrices')
        

