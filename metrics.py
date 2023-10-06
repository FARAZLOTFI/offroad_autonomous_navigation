from ignite.metrics import Precision, Recall, Accuracy, MeanSquaredError
import os
import math
from matplotlib import pyplot as plt
import numpy as np 
import torch

LOG_SIG_MAX = 0.5 
LOG_SIG_MIN = -0.5

class Metrics: 

    #initialize metrics
    def __init__(self, planning_horizon, device):
        self.planning_horizon = planning_horizon
        self.device = device
        self.recall = []
        self.precision = []
        self.f1 = []
        self.accuracy = []
        self.mse = []
        for _ in range(planning_horizon):
            self.recall.append(Recall(average=True, device=self.device))
            self.precision.append(Precision(average=True, device=self.device))
            p = Precision(device=self.device, average=False)
            r = Recall(device=self.device, average=False)
            f1 = (p * r * 2 / (p + r))
            self.f1.append(f1)
            self.accuracy.append(Accuracy(device=self.device))
            self.mse.append(MeanSquaredError(device=self.device))

    #reset all metrics
    def reset(self): 
        for i in range(self.planning_horizon): 
            self.recall[i].reset()
            self.precision[i].reset()
            self.f1[i].reset()
            self.accuracy[i].reset()
            self.mse[i].reset()
        return
    
    #update all metrics with batch data
    def update(self, outputs, labels, gauss_out = 1):
        for i in range(self.planning_horizon): 
            #classification metrics
            self.precision[i].update((outputs[0][i],labels[0][i]))
            self.recall[i].update((outputs[0][i],labels[0][i]))
            self.accuracy[i].update((outputs[0][i],labels[0][i]))
            self.f1[i].update((outputs[0][i],labels[0][i]))

            #regression metrics
            if gauss_out:
                self.mse[i].update((outputs[1][i][:,0], labels[1][i]))
            else:
                self.mse[i].update((outputs[1][i], labels[1][i]))
        return

    def kl_div(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2
        tr_term = (Sigs[:,None,:,:]*(Sigs**-1))#.sum(3)
        det_term = torch.log((Sigs/Sigs[:,None,:,:]))#.prod(3))
        #quad_term = torch.einsum('ijkl->ijk',(mus - mus[:,None,:,:])**2/Sigs)
        quad_term = ((mus - mus[:,None,:,:])**2/Sigs)
        #return .5 * (tr_term + det_term + quad_term - mus.shape[2])
        return .5 * (tr_term + det_term + quad_term - 1)

    def bhatt_div(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2
        mean_sig = (Sigs[:,None,:,:]+Sigs)/2
        #quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2/mean_sig)
        quad_term = (mus[:,None,:,:] - mus)**2/mean_sig
        log_term = torch.log(((mean_sig)/
                torch.sqrt(Sigs[:,None]*Sigs)))#.prod(3))
        return ((1/8)*quad_term+(1/2)*log_term)

    ## NOTE Lucas' funky fresh code
    ## please ask if you have questions
    """
    def wasserstein_dist(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2
        mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
        Sigs = Sigs.reshape(mus.shape[0], mus.shape[1], -1)
        sigs = sigs.reshape(mus.shape[0], mus.shape[1], -1)
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2)
        tr_term = (Sigs[:,None,:,:]+Sigs-
                2*torch.sqrt(sigs[:,None,:,:]*Sigs*sigs[:,None,:,:])).sum(3)
        return quad_term+tr_term


    def wasserstein_dist_zero_std(self, mus, unc_per_pixel=False):
        mus = mus.type(torch.float64)
        mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2)
        return quad_term
    """

    def pairwise_exp(self, mus, sigs, measure, numb_comp, device='cpu'):
        if measure == 'KL':
            dist = self.kl_div(mus, sigs)
        elif measure == 'Bhatt':
            dist = self.bhatt_div(mus, sigs)
        elif measure == 'Wass':
            dist = self.wasserstein_dist(mus, sigs)
        elif measure == 'Wass_0':
            dist = self.wasserstein_dist_zero_std(mus)
        weight = torch.tensor([1/numb_comp]).to(device).type(torch.float64)
        pairwise_dist = torch.log(weight)+weight*torch.log(torch.exp(-dist).sum(1)).sum(0)
        return -pairwise_dist, dist



    def calc_unc(self, model, inputs):
        classification_outputs = []
        regression_outputs = []
        for i in range(model.ensemble_size):
            model_outputs = model.training_phase_output(inputs, ensemble_comp=i)
            classification_outputs.append(model_outputs[0].detach().cpu())
            regression_outputs.append(model_outputs[1].detach().cpu())
        classification_outputs = torch.stack(classification_outputs)
        regression_outputs = torch.stack(regression_outputs)
        dist_metrics = ['KL', 'Bhatt']
        epi_unc_regressions = {}
        for dist in dist_metrics:
            sigs = torch.exp(torch.clamp(regression_outputs[:,:,:,-1], min=LOG_SIG_MIN, max=LOG_SIG_MAX))
            epi_unc, _ = self.pairwise_exp(regression_outputs[:,:,:,0], sigs, dist, model.ensemble_size)
            epi_unc_regressions[dist]=epi_unc
        cat_dist = torch.nn.functional.softmax(classification_outputs,dim=3)
        mix_dist = cat_dist.mean(0)
        cat_dist = cat_dist.type(torch.float64)
        mix_dist = mix_dist.type(torch.float64)
        pred_classification = mix_dist[0,1,:].argmax()
        pred_regression = regression_outputs[:,:,:,0].mean(0)
        epi_unc_classification = -(mix_dist*torch.log(mix_dist)).sum(2)+(cat_dist*torch.log(cat_dist)).sum(3).mean(0)
        return pred_classification, pred_regression, epi_unc_classification, epi_unc_regressions 
        

    #compute metrics with data passed in through update
    def compute(self, regression_filename=None, classification_filename=None, classification_title='Classification Metrics', regression_title='Regression Metrics'):
        p = []
        r = []
        f1 = []
        acc = []
        x = []
        mse = []
        for i in range(self.planning_horizon):
            x.append(i + 1)
            p.append(self.precision[i].compute())
            r.append(self.recall[i].compute())
            acc.append(self.accuracy[i].compute())
            mse.append(self.mse[i].compute())
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
    
    def plot_unc(self, epi_unc_clf, epi_unc_rg, ensemble_type, ensemble_size, path, epoch, seq_encoder):
        title_size = 15
        line_width = 4
        tick_size = 14
        color_hexes = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',
            '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharey=True, sharex=True)
        ax2 = ax.twinx()
        planning_horizon = epi_unc_clf.shape[0]
        epi_unc_clf = epi_unc_clf.cpu().detach().numpy()
        epi_unc_rg['KL'] = epi_unc_rg['KL'].cpu().detach().numpy()
        epi_unc_rg['Bhatt'] = epi_unc_rg['Bhatt'].cpu().detach().numpy()
        clf_line= ax.plot(np.arange(planning_horizon)+1, epi_unc_clf.mean(1),
                    c=color_hexes[0], linewidth=line_width, label = 'Clf_Unc')
        #clf_line= ax.plot(np.arange(planning_horizon)+1, (epi_unc_clf.mean(1)-epi_unc_clf.mean(1).min())/
        #            (epi_unc_clf.mean(1).max()-epi_unc_clf.mean(1).min()),
        #            c=color_hexes[0], linewidth=line_width, label = 'Clf_Unc')
        ax2.set(ylabel='')
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax2.set_yticks([])
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax2.tick_params(axis='y', labelsize=tick_size)
        kl_line = ax2.plot(np.arange(planning_horizon)+1, epi_unc_rg['KL'].mean(1),
                    c=color_hexes[1], linewidth=line_width, label = 'KL')
        bhatt_line = ax2.plot(np.arange(planning_horizon)+1, epi_unc_rg['Bhatt'].mean(1),
                    c=color_hexes[2], linewidth=line_width, label = 'Bhatt')
        #ax2.plot(np.arange(planning_horizon)+1, (epi_unc_rg['KL'].mean(1)-epi_unc_rg['KL'].mean(1).min())/
        #    (epi_unc_rg['KL'].mean(1).max()-epi_unc_rg['KL'].mean(1).min()),
        #    c=color_hexes[1], linewidth=line_width, label = 'KL')
        #ax2.plot(np.arange(planning_horizon)+1, (epi_unc_rg['Bhatt'].mean(1)-epi_unc_rg['Bhatt'].mean(1).min())/
        #    (epi_unc_rg['Bhatt'].mean(1).max()-epi_unc_rg['Bhatt'].mean(1).min()),
        #    c=color_hexes[2], linewidth=line_width, label = 'Bhatt')
        ax.set_title('Epistemic Uncertainty', fontdict={'fontsize':title_size})
        lns = clf_line+kl_line+bhatt_line
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        fig.tight_layout()
        unc_path = os.path.join(path, 'uncertainty')
        if not os.path.exists(unc_path):
            os.makedirs(unc_path)
        plt.savefig(os.path.join(unc_path, f'{ensemble_type}_{ensemble_size}_epoch{epoch}_{seq_encoder}_unc.png'))
        plt.close()
        np.save(os.path.join(unc_path, f'uncs_reg_{seq_encoder}_epoch{epoch}_{ensemble_type}_{ensemble_size}.npy'), epi_unc_rg)
        np.save(os.path.join(unc_path, f'uncs_clf_{seq_encoder}_epoch{epoch}_{ensemble_type}_{ensemble_size}.npy'), epi_unc_clf)
