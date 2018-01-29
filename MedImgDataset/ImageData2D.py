from torch.utils.data import Dataset
from torch import from_numpy, cat
import fnmatch
import os
import numpy as np
from scipy.ndimage import imread

class ImageDataSet2D(Dataset):
    def __init__(self, rootdir, readmode='L', verbose=False, dtype=float):
        """ImageDataSet2D
        Description
        -----------
          This class read 2D images with png/jpg file extension in the specified folder into torch tensor.

        :param str rootdir:  Specify which directory to look into
        :param str readmode: This argument is passed to @scipy.ndimage.imread
        :param bool verbose: Set to True if you want verbose info
        :param type dtype:   The type to cast the tensors
        """
        super(ImageDataSet2D, self).__init__()
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self.readmode=readmode
        self._ParseRootDir()

    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all .png, .jpg images to cache
        :return:
        """

        filenames = os.listdir(self.rootdir)
        filenames.sort()
        [self.dataSourcePath.extend(fnmatch.filter(filenames, "*" + ext)) for ext in ['.png','.jpg']]
        self.dataSourcePath = [self.rootdir + "/" + F for F in self.dataSourcePath]
        self.dataSourcePath.sort()

        for f in self.dataSourcePath:
            if self.verbose:
                print "Reading from ", f
            self.data.append(from_numpy(np.array(imread(f, mode=self.readmode), dtype=self.dtype)))

        self.length = len(self.data)
        # self.data = cat(self.data, dim=0).squeeze().contiguous()
        # self.length = self.data.size()[0]
        # print self.data.size()


    def __getitem__(self, item):
        return self.data[item]


    def __str__(self):
        from pandas import DataFrame as df
        s = "==========================================================================================\n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Image Details:\n" \
            "--------------\n"%(self.rootdir, self.length)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        printable = {'File Name': [], 'Size': []}
        for i in xrange(self.length):
            printable['File Name'].append(os.path.basename(self.dataSourcePath[i]))
            # for keys in self.metadata[i]:
            #     if not printable.has_key(keys):
            #         printable[keys] = []
            #
            #     printable[keys].append(self.metadata[i][keys])
            printable['Size'].append([self.__getitem__(i).size()[0],
                                      self.__getitem__(i).size()[1]])

        data = df(data=printable)
        s += data.to_string()
        return s

    def __len__(self):
        return self.length