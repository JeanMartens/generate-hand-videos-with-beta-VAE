import numpy as np
import pandas as pd 
import os
from jarviscloud import jarviscloud
from torch import nn
import torch

from train_valid import ModelTrainer
from fastnumpyio import fastnumpyio
from losses import *
from hyperparams import Hyperparams
from models import create_model

from stratified_kfold_loaders import *
from torch.cuda.amp import autocast

#Mock metadata
metadata = pd.read_csv('data/metadatas/metadata.csv')
all_images = np.load('data/images_labels/images.npy')

train_loaders, valid_loaders, splits_as_list = kfold_loaders(
        metadata = metadata, 
        all_images = all_images,
        normalise_transform = Hyperparams.normalise_transform,
        batch_size_train = Hyperparams.batch_size_train, 
        batch_size_valid = Hyperparams.batch_size_valid,
        num_splits=Hyperparams.num_splits, 
        random_state=Hyperparams.random_state)

criterion = nn.MSELoss()
# criterion = nn.L1Loss()

training_instance = ModelTrainer(create_model, train_loaders,valid_loaders,criterion)


if __name__ == "__main__":
    training_instance.execute(Hyperparams.num_epochs, 
                              splits_to_train=Hyperparams.splits_to_train)
