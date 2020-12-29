# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:19:32 2020

@author: Iacopo
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from CERTIFAI import CERTIFAI
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping

patience = 10
max_epochs = 200

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    strict = False,
    verbose = False,
    mode = 'min')

class Classifier(pl.LightningModule):
    def __init__(self, in_feats = 5, h_size = 25, out = 5, n_layers = 1,
                 activation_function = nn.ReLU, lr = 1e-3):
        super().__init__()
        
        mlp_sequence = OrderedDict([('mlp1', nn.Linear(in_features = in_feats,
                                                       out_features = h_size)),
                                    ('activ1', activation_function())])
        
        for i in range(1,n_layers):
            new_keys = ['mlp'+str(i+1), 'activ'+str(i+1)]
            mlp_sequence[new_keys[0]] = nn.Linear(in_features = h_size,
                                             out_features = h_size)
            mlp_sequence[new_keys[1]] = activation_function()
        
        mlp_sequence['out_projection'] = nn.Linear(in_features = h_size,
                                                   out_features = out)
        
        self.net = nn.Sequential(mlp_sequence)
        
        self.learning_rate = lr
    
    def forward(self, x, apply_softmax = False):
        y = self.net(x)
        if apply_softmax:
            return nn.functional.softmax(y, -1)
        return y
    
    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr = self.learning_rate)
        
        return optimiser
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = F.cross_entropy(yhat, y)
        self.log('val_loss', loss)
    
url = 'drug200.csv'

cert = CERTIFAI.from_csv(url)

model_input = cert.transform_x_2_input(cert.tab_dataset, pytorch = True)

target = model_input[:,-1].long()

cert.tab_dataset = cert.tab_dataset.iloc[:,:-1]

predictors = model_input[:,:-1]

val_percentage = 0.1

batch_size = 8

train_loader = DataLoader(
    TensorDataset(predictors[:-int(len(predictors)*val_percentage)],
                  target[:-int(len(predictors)*val_percentage)]),
    batch_size = batch_size)

val_loader = DataLoader(
    TensorDataset(predictors[-int(len(predictors)*val_percentage):],
                  target[int(-len(predictors)*val_percentage):]),
    batch_size = batch_size)

trainer = pl.Trainer(max_epochs = max_epochs, callbacks=[early_stop])

model = Classifier()

trainer.fit(model, train_loader, val_loader)

cert.fit(model, generations = 10, verbose = True)

print("(Unnormalised) model's robustness:")
print(cert.check_robustness(), '\n')
print("(Normalised) model's robustness:")
print(cert.check_robustness(normalised = True), '\n')
print("Model's robustness for male subgroup vs. female subgroup (to check fairness of the model):")
print(cert.check_fairness([{'Sex':'M'},{'Sex':'F'}]), '\n')
print("Visualising above results:")
print(cert.check_fairness([{'Sex':'M'},{'Sex':'F'}], visualise_results = True), '\n')
print("Obtain feature importance in the model and visualise:")
print(cert.check_feature_importance(visualise_results = True))