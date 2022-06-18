from torch.utils.data import  Dataset
import cv2


class CelebA_Dataset(Dataset):

    def __init__(self, dataset_txt, transforms = None):

        super().__init__()


        self.dataset_txt = dataset_txt
        self.tranforms = transforms

        with open(self.dataset_txt, "r") as file:

            self.images = file.read().rstrip("\n").split("\n")

    
    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_file = self.images[index]
        image = cv2.imread(image_file)
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.tranforms:

            image = self.tranforms(image)

        return image
