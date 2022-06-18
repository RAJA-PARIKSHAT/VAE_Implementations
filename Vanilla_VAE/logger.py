import os
import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
class Logger:

    def __init__(self, logging_dir, log_file_name, checkpoint_freq = None, ckpt_dir = None, visualization_freq = None, viz_dir = None):

        self.logging_dir = logging_dir
        self.loss_list = []
        self.checkpoint_freq = checkpoint_freq
        self.visualization_freq = visualization_freq
        self.ckpt_dir = ckpt_dir
        self.viz_dir = viz_dir

        if not os.path.exists(self.logging_dir):

            os.makedirs(self.logging_dir)

        
        self.log_file = open(os.path.join(self.logging_dir, log_file_name), 'a')
        self.names = None
        self.epoch = 0
    
    
    def log_iter(self, losses):

        if self.names is None:
            self.names = list(losses.keys())

        loss_values = [loss.detach().cpu().numpy() for loss in losses.values()]
        self.loss_list.append(loss_values)


    def log_epoch(self, epoch, model = None):

        self.epoch = epoch
        self.log_scores()

    def log_scores(self):

        loss_mean = np.array(self.loss_list).mean(axis = 0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.names, loss_mean)])
        loss_string = "Epoch " + str(self.epoch).zfill(5) + ": " + loss_string

        print(loss_string, file= self.log_file)
        self.loss_list = []
        self.log_file.flush()
    
    def save_ckpt(self, model):

        self.model = model
        if not os.path.exists(self.ckpt_dir):

            os.makedirs(self.ckpt_dir)

        cpk_path = os.path.join(self.ckpt_dir, '%s-checkpoint.pth.tar' % str(self.epoch + 1).zfill(5))

        torch.save(self.model.state_dict(), cpk_path)


    def visualize(self, reconstructed, generated):

        if not os.path.exists(self.viz_dir):

            os.makedirs(self.viz_dir)

        
        reconstructed = (reconstructed.detach().cpu())*255.
        generated = (generated.detach().cpu())*255.

        reconstructed_grid = make_grid(reconstructed, nrow= 8).permute(1,2,0).squeeze().numpy()
        generated_grid = make_grid(generated, nrow= 8).permute(1,2,0).squeeze().numpy()

        cv2.imwrite(os.path.join(self.viz_dir,"epoch_" + str(self.epoch + 1).zfill(5) + "reconstructed.jpg"), reconstructed_grid)
        cv2.imwrite(os.path.join(self.viz_dir,"epoch_" + str(self.epoch + 1).zfill(5) + "generated.jpg"), generated_grid)
        








