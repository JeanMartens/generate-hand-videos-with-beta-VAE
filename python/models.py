import timm
import torch 
import torch.nn as nn

from hyperparams import Hyperparams
import model_classes as mc

def create_model(accelerate):
    
    # model = timm.create_model(Hyperparams.encoder_name ,in_chans=1,num_classes = 10 ,pretrained=True)
    model = mc.AutoEncoder(encoder_name = Hyperparams.encoder_name, latent_dim = 2)
    model = model.float()
    
    return accelerate.prepare(model)

def model_naming_function(metric_score, epoch, Hyperparams):
    return f'me_{metric_score:.3f}_ep_{epoch}_en_{Hyperparams.encoder_name}_lr_{Hyperparams.lr}_si_{Hyperparams.img_shape[0]}.pt'.replace(",", "" )