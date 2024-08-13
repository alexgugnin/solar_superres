from torch.utils.data import Dataset
from typing import List
from PIL import Image 
import torch

class SolarDataset(Dataset):
  def __init__(self, targ_dir, res, transform=None) -> None:
    self.dirname = targ_dir
    self.res = res
    self.lowres: List = sorted(list((self.dirname / f"low_res_{self.res}/").glob('*.pt')))
    self.highres: List = sorted(list((self.dirname / "highres/").glob('*.pt')))
    self.transform = transform

  def load_image(self, path_list: List, index: int,) -> Image.Image:
    '''
    Opens an image via a path and returns it.
    '''
    return Image.open(path_list[index])

  def load_tensor(self, path_list: List, index: int) -> torch.Tensor:
    '''
    Opens a tensor via a path and returns it.
    '''
    return torch.load(path_list[index])
  
  def __len__(self) -> int:
    '''
    Returns the total number of samples.
    '''
    return len(self.lowres)

  def __getitem__(self, index: int):
    '''
    Returns one sample of data, lowres(input) and
    highres(target) versions of image (x, Y).
    '''
    input = self.load_tensor(self.lowres, index).float()
    target = self.load_tensor(self.highres, index).float()

    # Transform if necessary
    if self.transform:
      #return self.transform(input)[0].unsqueeze(dim=0), self.transform(target)[0].unsqueeze(dim=0) #[0] and unsqueeze for 1 wavelength only
      return self.transform(input), self.transform(target)
    else:
      return input, target