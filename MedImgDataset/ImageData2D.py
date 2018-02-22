from torch.utils.data import Dataset
from torch import from_numpy, cat
import fnmatch
import os
import numpy as np
from skimage.io import imread

class ImageDataSet2D(Dataset):
    def __init__(self, rootdir, as_grey=True, verbose=False, dtype=float, readfunc=None):
        """ImageDataSet2D
        Description
        -----------
          This class read 2D images with png/jpg file extension in the specified folder into torch tensor.

        :param str rootdir:  Specify which directory to look into
        :param str readmode: This argument is passed to skimage.io.imread
        :param bool verbose: Set to True if you want verbose info
        :param callable readfunc: If this is set, it will be used to load image files, as_grey option will be ignored
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
        self.as_grey=as_grey
        self.readfunc = readfunc
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
            if self.readfunc is None:
                im = imread(f, as_grey=self.as_grey)
            else:
                im = self.readfunc(f)
            im = from_numpy(np.array(im*255, dtype=self.dtype)) if self.as_grey and self.dtype == np.uint8 else \
                from_numpy(np.array(im, dtype=self.dtype))
            self.data.append(im)

        self.length = len(self.data)


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

    def tonumpy(self):
        assert self.length != 0
        return cat([K.unsqueeze(0) for K in self.data], dim=0).numpy()

if __name__ == '__main__':
    import visdom as vis

    v = vis.Visdom(port=80)

    data = ImageDataSet2D("./TOCI/10.TestData/Resized_SAR", dtype=np.float, verbose=True, as_gray=True)
    im = np.array([d.numpy()*255 for d in data.data], dtype=np.uint8)
    im = np.tile(im[:,None,:,:], (1, 3, 1, 1))
    v.images(im, env="Test", win="Image")