from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from pandas.core.common import flatten
import random



class HymenopteraDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.classes =[]
        for data_path in glob(root_dir + '/*'):
          self.classes.append(data_path.split('/')[-1])
          self.image_paths.append(glob(data_path + '/*.jpg'))

        self.image_paths = list(flatten(self.image_paths))
        random.shuffle(self.image_paths)

        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)

        # image = torch.stack([image] * 3, dim=0)
        # Assuming that the image name contains 'ant' or 'bee' to determine the label
        label = img_name.split('/')[-2]
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label



