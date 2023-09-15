import torch 
import torch.nn as nn 
import timm

class AutoEncoder(nn.Module):
    def __init__(self, encoder_name, latent_dim):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.encoder = timm.create_model(encoder_name,in_chans=1, pretrained=True,)
        num_embeddings  = self.encoder.classifier.in_features
        modules = list(self.encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        self.latent_layer = nn.Linear(num_embeddings, latent_dim)
        
        # decoder
        self.decoder_input = nn.Linear(latent_dim, num_embeddings)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_embeddings, 512, kernel_size=2, stride=2),  # 1x1 to 2x2
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 2x2 to 4x4
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 4x4 to 8x8
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 8x8 to 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  # 8x8 to 16x16
            nn.ReLU(),

            nn.Conv2d(1,1, kernel_size=5, stride=1, padding =0),
        )

        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent_layer(x)
        x = self.decoder_input(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        return x

    def encode(self,x): #expected shape : (BS, C, H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent_layer(x)
        return x

    def decode(self,x): #expected shape : (BS, latent_space)
        x = self.decoder_input(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        return x

