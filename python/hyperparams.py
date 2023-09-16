import albumentations as A
from torchvision import transforms
import numpy as np
import cv2

class Hyperparams:
    
    #Training Params
    lr = 5e-4
    num_epochs = 20
    batch_size_train = 256
    batch_size_valid = 256
    weight_decay = 0
    img_shape = (28,28,1)
    latent_space = 10


    #Model params
    encoder_name = 'efficientnet_b2'


    #Folds params
    num_splits = 5
    splits_to_train = [1]
    splits_to_oof = [1]
    
    random_state = 19

    normalise_transform = transforms.Compose([
        transforms.Normalize(mean=(159.5), std=(115.7570))
        ])

    augment_transform = A.Compose([
        # A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.RandomRotate90(),
    ], p=0.6)


