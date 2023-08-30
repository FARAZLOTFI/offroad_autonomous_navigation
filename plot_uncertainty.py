import os

import numpy as np
import matplotlib.pyplot as plt 

epoch = 800 
lstm_dir = '/home/nwaftp23/scratch/offroad_trainfiles_LSTM_26_08_23_10_17_10/uncertainty'
transformer_dir = '/home/nwaftp23/scratch/offroad_trainfiles_Transformer_26_08_23_10_17_10/uncertainty'

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
trans_reg_unc =np.load(os.path.join(transformer_dir, f'uncs_reg_Transformer_epoch{epoch}_fixed_masks_5.npy'),
    allow_pickle=True).item()
trans_clf_unc =np.load(os.path.join(transformer_dir, f'uncs_clf_Transformer_epoch{epoch}_fixed_masks_5.npy'),
    allow_pickle=True)
lstm_reg_unc =np.load(os.path.join(lstm_dir, f'uncs_reg_LSTM_epoch{epoch}_fixed_masks_5.npy'),
    allow_pickle=True).item()
lstm_clf_unc =np.load(os.path.join(lstm_dir, f'uncs_clf_LSTM_epoch{epoch}_fixed_masks_5.npy'),
    allow_pickle=True)
x = np.arange(lstm_clf_unc.shape[0])+1
ax[0].plot(x, trans_clf_unc.mean(1), label='Transformer')
ax[0].plot(x, lstm_clf_unc.mean(1), label='LSTM')
ax[0].set_title("Classification Uncertainty")
ax[0].legend()
ax[1].plot(x, trans_reg_unc['KL'].mean(1), label='Transformer_KL')
ax[1].plot(x, lstm_reg_unc['KL'].mean(1), label='LSTM_KL')
ax[1].plot(x, trans_reg_unc['Bhatt'].mean(1), label='Transformer_Bhatt')
ax[1].plot(x, lstm_reg_unc['Bhatt'].mean(1), label='LSTM_Bhatt')
ax[1].set_title("Regression Uncertainty")
ax[1].set_yscale('log')
ax[1].legend()
fig.tight_layout()
plt.savefig(f'/home/nwaftp23/scratch/unc_lstm_vs_transformer_epoch{epoch}.png')
plt.close()
'''
for i in range(500):
    epoch = i 

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    trans_reg_unc =np.load(
        f'/home/nwaftp23/scratch/offroad_trainfiles/uncs_reg_Transformer_epoch{epoch}_fixed_masks_5.npy',
        allow_pickle=True).item()
    trans_clf_unc =np.load(
        f'/home/nwaftp23/scratch/offroad_trainfiles/uncs_clf_Transformer_epoch{epoch}_fixed_masks_5.npy',
        allow_pickle=True)
    lstm_reg_unc =np.load(
        f'/home/nwaftp23/scratch/offroad_trainfiles/uncs_reg_LSTM_epoch{epoch}_fixed_masks_5.npy',
        allow_pickle=True).item()
    lstm_clf_unc =np.load(
        f'/home/nwaftp23/scratch/offroad_trainfiles/uncs_clf_LSTM_epoch{epoch}_fixed_masks_5.npy',
        allow_pickle=True)
    x = np.arange(lstm_clf_unc.shape[0])+1
    ax[0].plot(x, trans_clf_unc.mean(1), label='Transformer')
    ax[0].plot(x, lstm_clf_unc.mean(1), label='LSTM')
    ax[0].set_title("Classification Uncertainty")
    ax[0].legend()
    ax[1].plot(x, trans_reg_unc['KL'].mean(1), label='Transformer_KL')
    ax[1].plot(x, lstm_reg_unc['KL'].mean(1), label='LSTM_KL')
    ax[1].plot(x, trans_reg_unc['Bhatt'].mean(1), label='Transformer_Bhatt')
    ax[1].plot(x, lstm_reg_unc['Bhatt'].mean(1), label='LSTM_Bhatt')
    ax[1].set_title("Regression Uncertainty")
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(f'/home/nwaftp23/scratch/unc_lstm_vs_transformer_epoch{epoch}.png')
    plt.close()'''
