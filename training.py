from __future__ import print_function, division


import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def count_params(model, want=False):
    """Count the number of trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if want: 
        return total_params
    else:
        print("Number of trainable params =", total_params)
        return None

    
    
def get_lr(optimizer):
    """Fetch learning rate from optimizer"""
    return optimizer.state_dict()['param_groups'][0]['lr']
    
    
    
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    """
    Train a pytorch model an keep the best, as measured by validation score
    
    Parameters:
    ------------
    
        model : pytorch model
        
        criterion : loss function
        
        optimizer : torch.optim optimizer object
        
        scheduler : learning rate scheduler
        
        num_epochs : integer, default: 25
            Number of epochs to train model
            
    Returns:
    --------
    
        Best model as measured by validation score
    
    """
    since = time.time()

    # Keep track of histories
    history = {'train': {'loss': [], 'acc': []}, 
               'val':   {'loss': [], 'acc': []}}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            for metric, loss in zip(
                ('loss', 'acc'), (epoch_loss, epoch_acc)):
                history[phase][metric].append(loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

