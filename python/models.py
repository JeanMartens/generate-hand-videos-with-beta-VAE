import timm
import torch 
import torch.nn as nn

from hyperparams import Hyperparams
import model_classes as mc

def create_model(accelerate):
    
    # model = mc.AutoEncoder(encoder_name = Hyperparams.encoder_name, latent_dim = Hyperparams.latent_space)
    # model = mc.VariationalAutoEncoder(encoder_name = Hyperparams.encoder_name, latent_dim = Hyperparams.latent_space)
    model = mc.VanillaVariationalAutoEncoder(num_embeddings = 1024, latent_dim = Hyperparams.latent_space)
    model = model.float()
    
    return accelerate.prepare(model)

def model_naming_function(metric_score, epoch, Hyperparams):
    return f'me_{metric_score:.3f}_ep_{epoch}_en_{Hyperparams.encoder_name}_lr_{Hyperparams.lr}_si_{Hyperparams.img_shape[0]}.pt'.replace(",", "" )