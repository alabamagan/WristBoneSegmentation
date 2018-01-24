from torch.utils.data import Dataset
from torch import from_numpy
import fnmatch
import os
import numpy as np
import SimpleITK as sitk

class ImageDataSet2D(Dataset):
    def __init__(self, rootdir, verbose=False, dtype=float):
        super(ImageDataSet2D, self).__init__()
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self._ParseRootDir()

    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all .png, .jpg images to cache
        :return:
        """