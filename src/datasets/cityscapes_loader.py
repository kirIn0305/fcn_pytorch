import numpy as np
import glob
import os
from torch.utils import data


class CityspcapesLoader(data.Dataset):
    """City Scapes dataset Loader
        caution! : this loader load only fine data, not Coarse.
    """
    def __init__(self, data_dir="../data/processed/label_00"):
        self.data_dir = data_dir
        self.data_fnames = self.get_fnames

    def get_fnames(self, data_dir):
        img_list = glob.glob(os.path.join(data_dir, "*_leftImg8bit"))
        data = [[i, i.replace("leftImg8bit", "gtFine_color")] for i in img_list]
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_fname, lbl_fname = self.data_fnames[index]
        img = np.load(img_fname)
        lbl = np.load(lbl_fname)
        return img, lbl

if __name__ == '__main__':
    cityscape_loader = CityspcapesLoader()
    for img, lbl in cityscape_loader:
        print(img)
        print(lbl)
