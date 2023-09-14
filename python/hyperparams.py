import albumentations as A
from torchvision import transforms
import numpy as np
import cv2

class Hyperparams:
    
    #Training Params
    lr = 1e-4
    num_epochs = 10
    batch_size_train = 256
    batch_size_valid = 256
    weight_decay = 1e-5
    img_shape = (28,28,1)


    #Model params
    encoder_name = 'efficientnet_b0'


    #Folds params
    num_splits = 5
    splits_to_train = [1]
    splits_to_oof = [1]
    
    random_state = 19

    normalise_transform = transforms.Compose([
        transforms.Normalize(mean=(0.0), std=(1.0))
        ])

    augment_transform = A.Compose([
        # A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.RandomRotate90(),
    ], p=0.6)


