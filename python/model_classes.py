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
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 2x2 to 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 4x4 to 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 8x8 to 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),  # 16x16 to 32x32
            nn.ReLU(),
        
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),
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

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder_name, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        
        # encoder
        self.encoder = timm.create_model(encoder_name, in_chans=1, pretrained=True,)
        num_embeddings = self.encoder.classifier.in_features
        modules = list(self.encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        
        # Producing mean and log variance
        self.mu_layer = nn.Linear(num_embeddings, latent_dim)
        self.logvar_layer = nn.Linear(num_embeddings, latent_dim)
        
        # decoder
        self.decoder_input = nn.Linear(latent_dim, num_embeddings)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_embeddings, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.ReLU(),
        
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),
        )

        self.norm = torch.distributions.Normal(0, 1)

        
    def reparameterize(self, x_mu, x_logvar, training=True):
        x_std = torch.exp(x_logvar / 2)

        z = x_mu + x_std * self.norm.sample(x_mu.shape).to(x_mu.device)
        return z

        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        x_mu = self.mu_layer(x)
        x_logvar = self.logvar_layer(x)
        
        z = self.reparameterize(x_mu, x_logvar, training=self.training)
        
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        
        return x , x_mu, x_logvar

    def encode(self, x, get_stats = False ):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x_mu = self.mu_layer(x)
        x_logvar = self.logvar_layer(x)
    
        z = self.reparameterize(x_mu, x_logvar, training=self.training)
        
        if get_stats :
            return z, x_mu,  x_logvar
        else : 
            return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        return x

class VanillaVariationalAutoEncoder(nn.Module):
    def __init__(self, num_embeddings,latent_dim):
        super(VanillaVariationalAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # 32x32 to 28x28
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 28x28 to 14x14
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        
            # 14x14 to 7x7
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        
            # 7x7 to 3x3
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        
            # 3x3 to 1x1
            nn.Conv2d(512, num_embeddings, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Producing mean and log variance
        self.mu_layer = nn.Linear(num_embeddings, latent_dim)
        self.logvar_layer = nn.Linear(num_embeddings, latent_dim)
        
        # decoder
        self.decoder_input = nn.Linear(latent_dim, num_embeddings)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_embeddings, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.ReLU(),
        
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),
        )

        self.norm = torch.distributions.Normal(0, 1)

        
    def reparameterize(self, x_mu, x_logvar, training=True):
        x_std = torch.exp(x_logvar / 2)

        z = x_mu + x_std * self.norm.sample(x_mu.shape).to(x_mu.device)
        return z

        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        x_mu = self.mu_layer(x)
        x_logvar = self.logvar_layer(x)
        
        z = self.reparameterize(x_mu, x_logvar, training=self.training)
        
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        
        return x , x_mu, x_logvar

    def encode(self, x, get_stats = False ):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x_mu = self.mu_layer(x)
        x_logvar = self.logvar_layer(x)
    
        z = self.reparameterize(x_mu, x_logvar, training=self.training)
        
        if get_stats :
            return z, x_mu,  x_logvar
        else : 
            return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.decoder(x)
        return x
