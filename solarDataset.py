from torch.utils.data import Dataset
from typing import List
from PIL import Image

class SolarDataset(Dataset):
    '''
    Custom dataset for the solar data. Paths should be changed manually for
    any specific case.
    '''
    def __init__(self, targ_dir, transform=None) -> None:
        self.dirname = targ_dir
        self.lowres: List = sorted(list((self.dirname / "low_res_64/").glob('*.jpeg')))
        self.highres: List = sorted(list((self.dirname / "high_res/").glob('*.jpeg')))
        self.transform = transform

    def load_image(self, path_list: List, index: int,) -> Image.Image:
        '''
        Opens an image via a path and returns it.
        '''
        return Image.open(path_list[index])

    def __len__(self) -> int:
        '''
        Returns the total number of samples.
        '''
        return len(self.lowres)

    def __getitem__(self, index: int):
        '''
        Returns one sample of a data, lowres(input) and highres(target) versions
        of an image (x, Y). If transformation is needed (for example, basically
        convert into torch.Tensor) - returnes transformed images.
        '''
        input = self.load_image(self.lowres, index)
        target = self.load_image(self.highres, index)

        # Transform if transormation exists
        if self.transform:
            return self.transform(input), self.transform(target)
        else:
            return input, target
