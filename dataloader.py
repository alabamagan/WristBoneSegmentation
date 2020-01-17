import os
import fnmatch
import numpy as np
import random
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader


class BatchLoader(Dataset):
    def __init__(self, rootdir, tonumpy=False):
        self.root_dir = rootdir
        self._ParseRootdir()
        self.tonumpy = tonumpy
        self.directions = {}

    def __getitem__(self, index):
        if type(index) == tuple:
            idx, idy = index
        else:
            idx = index
            idy = None

        out = {}
        for keys in self.unique_samples[idx]:
            out[keys] = sitk.ReadImage(self.unique_samples[idx][keys])
            self.directions[index] = out[keys].GetDirection()
            if self.tonumpy:
                out[keys] = sitk.GetArrayFromImage(sitk.ReadImage(self.unique_samples[idx][keys]))
        return out



    def __call__(self, num):
        """
        Description
        -----------
          This function will return the specified amount of samples randomly drawn from the 
          rootdir. Only works in numpy mode. This method is suitable for training.
          
        :param: int num Number of random drawn samples
        :return:
        """

        assert self.tonumpy, "This method will only return numpy arrays"
        assert isinstance(num, int), "Call with index"
        assert num > 0, "Why would you want zero samples?"


        # Randomly select image indexes
        indexes = random.sample(np.arange(self.length), np.min([np.random.randint(1, 5), num]))
        sliceperimage = int(np.floor(num / float(len(indexes))))


        out = {'im': None, 'A': None, 'B': None, 'M': None}
        for i in indexes:
            ims = self.__getitem__(i)
            assert isinstance(ims['im'], np.ndarray), "Re-run _ParseRootdir() to re-initialize this object."

            # Check if last index
            if i == indexes[-1]:
                l_s = random.sample(np.argwhere(ims['A'].sum(1).sum(1)).flatten().tolist(), sliceperimage + num % len(indexes))
            else:
                l_s = random.sample(np.argwhere(ims['A'].sum(1).sum(1)).flatten().tolist(), sliceperimage)

            # Make sure that all layers has labels
            for keys in ['im', 'M']:
                RR = [ims[keys][j] for j in l_s]
                RR = [r.reshape(1, r.shape[0], r.shape[1]) for r in RR]
                if out[keys] is None:
                    out[keys] = RR
                else:
                    RR.extend(out[keys])
                    out[keys] = RR

        for keys in ['im', 'M']:
            out[keys] = np.concatenate(out[keys], 0)

        return out

    def _ParseRootdir(self):
        """
        Description
        -----------
          Process the root dir and identify unique sample

        :return:
        """

        def cmp(x, y):
            """
            A temporally compare function for sorting the filenames. Assume file name in format:
                [sample_number]_([A/B]_)[TYPE].nii.gz

            """
            assert isinstance(x, str) and isinstance(y, str), "Only for comparing filenames."

            X = int(x.split('_')[0])
            Y = int(y.split('_')[0])

            if (X < Y):
                return -1
            elif (X==Y):
                return 0
            else:
                return 1

        imageT1dir = "89_MRT1W"
        imagedir = "90_MRT2W"
        # imagedir = "98_MRT2W_FLIPED"
        # imagedir = "98_CARPAL_FLIPED"
        # segdir = "91_CARPAL"
        segdir = "92_CARPAL_ONELABEL"
        # segdir = "97_CARPAL_ONELABEL_FLIPED"
        # segdir = "93_CARPAL_Seperated"
        # segdir = "95_CARPAL_INVERSE"


        d = [self.root_dir + "/" + imageT1dir,
             self.root_dir + "/" + imagedir,
             self.root_dir + "/" + segdir]
        getfilenames = lambda x: fnmatch.filter(os.listdir(x), "*.nii.gz")

        # Load Images
        imageT1filenames = getfilenames(d[0])
        imagefilenames = getfilenames(d[1])
        segfilenames = getfilenames(d[2])
        self.unique_suffix = [n.split('/')[-1].split('_')[1] for n in segfilenames]
        self.unique_suffix = list(set(self.unique_suffix))

        imageT1filenames.sort(cmp=cmp)
        imagefilenames.sort(cmp=cmp)
        segfilenames.sort(cmp=cmp)

        # Sort images into array of dictionaries
        self.length = len(imagefilenames)
        self.unique_samples = []
        for i in range(self.length):
            ld = {}
            ld['im'] = self.root_dir + "/" + imagedir + "/" + imagefilenames[i]
            ld['imt1'] = self.root_dir + "/" + imageT1dir + '/' + imageT1filenames[i]

            for u in self.unique_suffix:
                ld[u] = self.root_dir + "/" + segdir + "/" + fnmatch.filter(segfilenames, "%i_%s_*"%(i + 1, u))[0]

            self.unique_samples.append(ld)

    def Draw3DBatch(self, batchsize, numberOfSlices):
        """
        Description
        -----------

        :param batchsize:
        :param numberOfSlices:
        :return: (B, C, Z, H, W)
        """

        read = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        indexes = random.sample(np.arange(self.__len__()), batchsize)
        suffix = ['im']
        suffix.extend(self.unique_suffix)

        # Load all images first
        images = {}
        # images['im'] = [read(self.unique_samples[i]['im']) for i in indexes]
        for k in suffix:
            images[k] = [read(self.unique_samples[i][k]) for i in indexes]

        # Get max number of slice for each image set
        s = [d.shape[0] for d in images['im']]
        s = [random.choice(np.arange(S - numberOfSlices - 1)) for S in s]
        e = [S + numberOfSlices for S in s]
        out = {}
        for k in suffix:
            out[k] = [images[k][i][s[i]:e[i]] for i in xrange(batchsize)]
            out_k_s = out[k][0].shape
            out[k] = [I.reshape(1, out_k_s[0], out_k_s[1], out_k_s[2]) for I in out[k]]
            out[k] = np.concatenate(out[k], axis=0)
        return out

    def DrawRandom3DPatchs(self, batchsize, volsize):
        """
        Description
        ------------

        :param numberOfPatchs:
        :param patchsize:
        :return:
        """

        assert self.tonumpy, "This method only available in np mode"

        patchsize = list(volsize)
        read = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
        suffix = ['im', 'imt1']
        suffix.extend(self.unique_suffix)

        out = {}
        for k in suffix:
            out[k] = None

        for i in range(batchsize):
            index = np.random.randint(0, self.__len__())
            ims = {}
            for k in suffix:
                ims[k] = read(self.unique_samples[index][k])
            d = ims['im'].shape
            max_r = list(np.array(d) - np.array(patchsize))
            start = [random.randint(0, mr) for mr in max_r]
            end = [start[j] + patchsize[j] for j in xrange(3)]

            for k in suffix:
                if(out[k] is None):
                    out[k] = np.copy(ims[k][start[0]:end[0], start[1]:end[1], start[2]:end[2]])
                    outshape = out[k].shape
                    out[k] = [out[k].reshape(1, 1, outshape[0], outshape[1], outshape[2])]
                else:
                    temp = np.copy(ims[k][start[0]:end[0], start[1]:end[1], start[2]:end[2]])
                    outshape = temp.shape
                    temp = temp.reshape(1, 1, outshape[0], outshape[1], outshape[2])
                    out[k].append(temp)

            del ims


        for k in suffix:
            out[k] = np.concatenate(out[k], axis=0)

        return out

    def DrawByCategories(self, num, csv, catagory):
        """
        Description
        -----------
            Read a csv file containing most of the
        :param int num:
        :param str csv:
        :param int catagory:
        :return:
        """

        import pandas as pd

        assert isinstance(num, int), "Batchsize must be integer!"
        assert os.path.isfile(csv), "Cannot find csv"

        # Read string
        def parse_category_string(str):
            s = str.split('_')
            out = []
            for pairs in s:
                if pairs.find('-') > -1:
                    out.extend(range(int(pairs.split('-')[0]), int(pairs.split('-')[1])+1))
                else:
                    out.append(int(pairs))
            return out

        data = pd.read_csv(csv)
        cat = data.columns[catagory]
        indexes = list(data.index)

        out = {'im': [], 'F': []}
        for i in xrange(num):
            index = random.choice(indexes)
            slice = parse_category_string(data[cat][index])
            slice = random.choice(slice)
            ims = self.__getitem__(index)
            for keys in out:
                im = ims[keys][slice - 1]
                im = im.reshape(1, im.shape[0], im.shape[1])
                out[keys].append(im)


        for keys in ['im', 'F']:
            out[keys] = np.concatenate(out[keys], 0)

        return out

    def LoadByCategories(self, item, csv, cat):
        """
        Description
        -----------
          Load images slices by category

        :param int item:
        :param str csv:
        :param int cat:
        :return:
        """

        assert isinstance(item, int), "Item must be integer!"
        assert os.path.isfile(csv), "Cannot find csv"

        def parse_category_string(str):
            s = str.split('_')
            out = []
            for pairs in s:
                if pairs.find('-') > -1:
                    out.extend(range(int(pairs.split('-')[0]), int(pairs.split('-')[1])+1))
                else:
                    out.append(int(pairs))
            return out

        im = self.__getitem__(item)


    def __len__(self):
        return self.length
