from distutils.log import Log
from pickletools import optimize

from cv2 import log
from utils import *
from dataset import CelebA_Dataset
from logger import Logger
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import transforms
import torch
from tqdm import tqdm


def train(nEpochs, ckpt_freq, viz_freq, model, train_loader, optimizer, validation_loader = None):


    train_logger = Logger(logging_dir= "logs", log_file_name= "train_logs.txt", checkpoint_freq= ckpt_freq, ckpt_dir= "checkpoints")
    
    if validation_loader:
        validation_logger = Logger(logging_dir="logs", log_file_name= "validation_logs.txt", visualization_freq= viz_freq,viz_dir= "visualizations")

    for epoch in tqdm(range(nEpochs)):

        model.train()
        for images in tqdm(train_loader):

            images = images.to(DEVICE)
            out_loss = model(images,DEVICE)
            optimizer.zero_grad()
            out_loss["loss"].backward()

            optimizer.step()

            train_logger.log_iter(losses= out_loss)
        
        train_logger.log_epoch(epoch= epoch)

        if (epoch + 1) % ckpt_freq == 0:
            train_logger.save_ckpt(model= model)

        if validation_loader:
            
            model.eval()

            with torch.no_grad():

                for i,images in enumerate(validation_loader):

                    images = images.to(DEVICE)
                    out_loss, reconstructed = model(images,DEVICE,True)
                    validation_logger.log_iter(losses= out_loss)

                    if (epoch + 1) % viz_freq == 0:

                        if i ==0:
                            noise = torch.randn(size = (batch_size, latent_dims)).to(DEVICE)
                            generated = model.get_sample_from_latent(noise)
                            validation_logger.visualize(reconstructed= reconstructed, generated= generated)
                
                validation_logger.log_epoch(epoch)
                

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("-epochs", "--NUMBER_OF_EPOCHS", required = False, default= 1000, help = "number of epochs to train")
    args.add_argument("-lr", "--LEARNING_RATE", required = False,default= 0.00025,  help = "learning rate for optimizer")
    args.add_argument("-latent_dims", "--LATENT_DIMS", required = False,default= 512, help = "latent space dimensions")
    args.add_argument("-ckpt_freq", "--CHECKPOINT_FREQ", required = False,default= 100, help = "checkpoints to create")
    args.add_argument("-viz_freq", "--VISUALIZATION_FREQ", required = False, default= 10, help = "visualizations")
    args.add_argument("-batch_size", "--BATCH_SIZE", required = False, default= 32)

    arguments = args.parse_args()
    options = vars(arguments)


    nEpochs = int(options["NUMBER_OF_EPOCHS"])
    learning_rate = float(options["LEARNING_RATE"])
    latent_dims = int(options["LATENT_DIMS"])
    ckpt_freq = int(options["CHECKPOINT_FREQ"])
    viz_freq  = int(options["VISUALIZATION_FREQ"])
    batch_size = int(options["BATCH_SIZE"])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    variational_autoencoder = VariationalAutoencoder(in_channels= 3, latent_dim= latent_dims, out_channels= 3)
    variational_autoencoder.to(DEVICE)
    
    train_transforms = transforms.Compose(transforms=[
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p = 0.3),
    transforms.ToTensor()        
    ])

    training_data = CelebA_Dataset(dataset_txt= "dataset_txts/train.txt", transforms= train_transforms)
    validation_data = CelebA_Dataset(dataset_txt= "dataset_txts/valid.txt", transforms= transforms.ToTensor())

    train_loader = DataLoader(dataset= training_data, batch_size= batch_size, shuffle= True, num_workers= 4)
    validation_loader = DataLoader(dataset= validation_data, batch_size= 32, shuffle= True, num_workers= 4)

    optimizer = Adam(variational_autoencoder.parameters(), lr= learning_rate)

    train(nEpochs= nEpochs, ckpt_freq= ckpt_freq, viz_freq= viz_freq, model= variational_autoencoder, train_loader= train_loader, optimizer= optimizer, validation_loader= validation_loader)


