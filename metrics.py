from ignite.metrics import Precision, Recall, Accuracy
import math
from matplotlib import pyplot as plt

class Metrics: 

    #initialize metrics
    def __init__(self, planning_horizon, device):
        self.planning_horizon = planning_horizon
        self.device = device
        self.recall = []
        self.precision = []
        self.f1 = []
        self.accuracy = []
        for _ in range(planning_horizon):
            self.recall.append(Recall(average=True, device=self.device))
            self.precision.append(Precision(average=True, device=self.device))
            p = Precision(device=self.device, average=False)
            r = Recall(device=self.device, average=False)
            f1 = (p * r * 2 / (p + r)).mean()
            self.f1.append(f1)
            self.accuracy.append(Accuracy(device=self.device))

    #reset all metrics
    def reset(self): 
        for i in range(self.planning_horizon): 
            self.recall[i].reset()
            self.precision[i].reset()
            self.f1[i].reset()
            self.accuracy[i].reset()
        return
    
    #update all metrics with batch data
    def update(self, outputs, labels):
        for i in range(self.planning_horizon): 
            self.precision[i].update((outputs[i],labels[i]))
            self.recall[i].update((outputs[i],labels[i]))
            self.accuracy[i].update((outputs[i],labels[i]))
            self.f1[i].update((outputs[i],labels[i]))
        return
    
    #compute metrics with data passed in through update
    def compute(self, filename=None):
        p = []
        r = []
        f1 = []
        acc = []
        for i in range(self.planning_horizon):
            p.append(self.precision[i].compute())
            r.append(self.recall[i].compute())
            acc.append(self.accuracy[i].compute())
            f1.append(self.f1[i].compute().item())   
        f1 = [0 if math.isnan(x) else x for x in f1]
        
        print("---METRICS---\nPrecision: {}\nRecall: {}\nF1: {}\nAccuracy: {}\n".format(p, r, f1, acc))

        if filename:
            plt.plot(p, label='Precision')
            plt.plot(r, label='Recall')
            plt.plot(acc, label='Accuracy')
            plt.plot(f1, label="F1")
            plt.xlabel('Planning Horizon')
            plt.ylabel('Score')
            plt.title('Metrics')
            plt.legend()
            plt.savefig(filename)
            plt.cla()
    

