from operator import mod
from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size


        if self.in_channels != self.out_channels:

            self.expand_channels = True
        
        else:
            self.expand_channels = False

        # All convolutional layers has their bias turned off, since batchnormalization is applied
        self.conv1 = nn.Conv2d(in_channels= self.in_channels, out_channels= self.hidden_channels, kernel_size= self.kernel_size, padding = "same", stride= 1, bias= False)
        self.conv2 = nn.Conv2d(in_channels= self.hidden_channels, out_channels= self.out_channels, kernel_size= self.kernel_size, padding = "same", stride= 1, bias= False)

        self.batch_norm1 = nn.BatchNorm2d(num_features= self.hidden_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features= self.out_channels)

        self.relu1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.relu2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        
        if self.expand_channels:

            self.conv3 = nn.Conv2d(in_channels= self.in_channels, out_channels= self.out_channels, kernel_size= 1, padding = 0, stride= 1, bias= False)

    
    def forward(self, inputs):

        if self.expand_channels:

            identity_mapping = self.conv3(inputs)

        else:
            identity_mapping = inputs

        output = self.conv1(inputs)
        output = self.batch_norm1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.batch_norm2(output)

        output = output + identity_mapping

        output = self.relu2(output)

        return output


class Encoder(nn.Module):

    def __init__(self, in_channels = 3, latent_dims = 512, channels = [32,64,128,256,512,512]):

        super().__init__()

        self.in_channels = in_channels
        self.latent_dims = latent_dims
        self.channels = channels
        
        self.encoder = nn.Sequential()

        first = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels= channels[0], kernel_size= 5, stride= 1, padding= 2, bias = False),
            nn.BatchNorm2d(num_features= channels[0]),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size= (2,2))
        )

        self.encoder.add_module("first",first)


        for i in range(1,len(channels)):

            residual_layer = nn.Sequential(
                ResidualBlock(in_channels= self.channels[i-1],hidden_channels= self.channels[i-1],out_channels= self.channels[i],kernel_size= 3),
                nn.AvgPool2d(kernel_size= (2,2))
            )

            self.encoder.add_module("resnet_block_{0}".format(i), residual_layer)

            self.latent_paramters = nn.Linear(self.channels[-1]*4*4, out_features= 2 * self.latent_dims)

    
    def forward(self, inputs):
        
        encoded =  self.encoder(inputs)
        encoded = encoded.view(inputs.shape[0], -1)

        latent_mu, latent_log_var = self.latent_paramters(encoded).chunk(2, dim = 1)

        return latent_mu, latent_log_var



class Decoder(nn.Module):


    def __init__(self, latent_dims = 512, out_channels = 3, channels = [512,512,256,128,64,32]):

        super().__init__()

        self.latent_dims = 512
        self.out_channels = 3
        self.channels = channels

        self.decoder = nn.Sequential()


        self.first = nn.Sequential(
            nn.Linear(in_features= self.latent_dims, out_features= channels[0]*4*4),
            nn.ReLU(inplace= True)
         )


        for i in range(1,len(channels)):

            residual_layer = nn.Sequential(
                ResidualBlock(in_channels= self.channels[i-1],hidden_channels= self.channels[i-1],out_channels= self.channels[i],kernel_size= 3),
                nn.Upsample(scale_factor= 2, mode= "nearest")
            )
            self.decoder.add_module("resnet_block_{0}".format(i), residual_layer)

        
        output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels= channels[-1], out_channels= self.out_channels, kernel_size= (4,4), stride= 2, padding= 1),
            nn.Sigmoid())
        self.decoder.add_module("output", output_layer)
    
    def forward(self, z):

        intermediate = self.first(z)
        intermediate = intermediate.view(z.shape[0], -1, 4, 4)
        
        output = self.decoder(intermediate)
        return output



class VariationalAutoencoder(nn.Module):

    def __init__(self,in_channels = 3, encoder_channels = [32,64,128,256,512,512] , latent_dim = 512, decoder_channels = [32,64,128,256,512,512], out_channels = 3):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        self.latent_dims = latent_dim


        self.encoder = Encoder(self.in_channels, self.latent_dims, self.encoder_channels)
        self.decoder = Decoder(self.latent_dims,self.out_channels ,self.decoder_channels)


    def reparameterize(self, latent_mean, latent_log_var, device):

        epsilon = torch.randn_like(latent_mean).to(device) 

        latent_z = latent_mean + torch.exp(latent_log_var / 2.)*epsilon

        return latent_z


    def get_variational_posterior(self, input):

        latent_mean, latent_log_var = self.encoder(input)

        return latent_mean, latent_log_var

    def get_sample_from_latent(self, latent):

        return self.decoder(latent)

    def forward(self, inputs,device, outputs = False):

        latent_mean, latent_log_var = self.get_variational_posterior(input= inputs)

        latent_z = self.reparameterize(latent_mean= latent_mean, latent_log_var= latent_log_var, device= device)

        reconstructed = self.get_sample_from_latent(latent_z)

        reconstruction_loss = F.mse_loss(input= inputs, target= reconstructed, reduction= "none").view(inputs.shape[0], -1).sum(dim= 1).mean()

        kl_divergence_loss = ( -0.5*(1 + latent_log_var -latent_mean**2  - torch.exp(latent_log_var) ) ).sum(dim = 1).mean()

        overall_loss = reconstruction_loss + kl_divergence_loss

        if not outputs:
            return OrderedDict(loss = overall_loss, recon_loss = reconstruction_loss, kl_loss = kl_divergence_loss)
        else:
            return OrderedDict(loss = overall_loss, recon_loss = reconstruction_loss, kl_loss = kl_divergence_loss), reconstructed
        






        




        
