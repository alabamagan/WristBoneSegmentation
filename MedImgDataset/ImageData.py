from torch.utils.data import Dataset
from torch import from_numpy, cat
import fnmatch
import os
import numpy as np
import SimpleITK as sitk
from imgaug import augmenters as iaa
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm

NIFTI_DICT = {
    "sizeof_hdr": int,
    "data_type": str,
    "db_name": str,
    "extents": int,
    "session_error": int,
    "regular": str,
    "dim_info": str,
    "dim": int,
    "intent_p1": float,
    "intent_p2": float,
    "intent_p3": float,
    "intent_code": int,
    "datatype": int,
    "bitpix": int,
    "slice_start": int,
    "pixdim": float,
    "vox_offset": float,
    "scl_slope": float,
    "scl_inter": float,
    "slice_end": int,
    "slice_code": str,
    "xyzt_units": str,
    "cal_max": float,
    "cal_min": float,
    "slice_duration": float,
    "toffset": float,
    "glmax": int,
    "glmin": int,
    "descrip": str,
    "aux_file": str,
    "qform_code": int,
    "sform_code": int,
    "quatern_b": float,
    "quatern_c": float,
    "quatern_d": float,
    "qoffset_x": float,
    "qoffset_y": float,
    "qoffset_z": float,
    "srow_x": str,
    "srow_y": str,
    "srow_z": str,
    "intent_name": str,
    "magic": str
}

class ImageDataSet(Dataset):
    """
    This dataset automatically load all the nii files in the specific directory to
    generate a 3D dataset
    """
    def __init__(self, rootdir, filelist=None, filesuffix=None, loadBySlices=-1, verbose=False, dtype=float, debugmode=False):
        """

        :param rootdir:
        """
        super(Dataset, self)
        assert os.path.isdir(rootdir), "Cannot access directory!"
        assert loadBySlices <= 2, "This class only handle 3D data!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.metadata = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self.filelist = filelist
        self.filesuffix = filesuffix
        self._debug=debugmode
        self._byslices=loadBySlices
        self.useCatagory = False
        self.loadCatagory = False
        self._ParseRootDir()


    def LoadWithCatagories(self, txtdir):
        """
        Writter for inception

        :param txtdir:
        :return:
        """
        import pandas as pd


        assert os.path.isfile(txtdir)


        def parse_category_string(str):
            s = str.split('_')
            out = []
            for pairs in s:
                if pairs.find('-') > -1:
                    out.extend(range(int(pairs.split('-')[0]), int(pairs.split('-')[1]) + 1))
                else:
                    out.append(int(pairs))
            return out

        self.catagory = {}
        cat = pd.read_csv(txtdir)
        for i, row in cat.iterrows():
            self.catagory[row['Name']] = [parse_category_string(row[row.keys()[i]]) for i in xrange(1,4)]

        availablelist = [int(os.path.basename(d).split('_')[0]) for d in self.dataSourcePath]
        temp = []
        for k,x in enumerate(availablelist):
            catlist = np.zeros(self.data[k].size(0))
            for i in xrange(3):
                for j in self.catagory[x][i]:
                    catlist[j - 1] = i
            temp.extend(catlist)
        temp = np.array(temp).flatten()

        data = cat(self.data, dim=0).numpy()
        self.data = np.concatenate([self.ResizeToSquare(data[i], 299)[None,:] for i in xrange(data.shape[0])])
        self.data = from_numpy(self.data)

        self._catagories = from_numpy(temp)
        self.length = self.data.size(0)
        self.loadCatagory = True
        pass

    def UseCatagories(self, txtdir, catagory=0):
        """

        :param txtdir:
        :return:
        """
        import pandas as pd


        assert os.path.isfile(txtdir)


        def parse_category_string(str):
            if str == 'NULL':
                return []
            try:
                s = str.split('_')
                out = []
                for pairs in s:
                    if pairs.find('-') > -1:
                        out.extend(range(int(pairs.split('-')[0]), int(pairs.split('-')[1])+1))
                    else:
                        try:
                            out.append(int(pairs))
                        except:
                            pass
                return out
            except:
                return []

        self.useCatagory = True
        self.catagory = {}

        cat = pd.read_csv(txtdir)
        for i, row in cat.iterrows():
            self.catagory[row['Name']] = [parse_category_string(row[row.keys()[i]]) for i in xrange(1,4)]
        availablelist = self.catagory.keys()
        availablelist.sort()
        temp = []
        for k,x in enumerate(availablelist):
            for i in xrange(len(self.catagory[x])):
                for j in self.catagory[x][i]:
                    temp.append([k,i + 1,j])

        self._itemindexes = np.array(temp) # [image index, catagory, slice number of]

        if catagory != 0:
            self._itemindexes = self._itemindexes[self._itemindexes[:,1] == catagory]
            self.length = len(self._itemindexes)

        pass


    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all nii images to cache

        :return:
        """

        if self.verbose:
            print "Parsing root path: ", self.rootdir

        # Load files written in filelist from the root_dir
        if self.filelist is None:
            filenames = os.listdir(self.rootdir)
            filenames = fnmatch.filter(filenames, "*.nii.gz")
        else:
            filelist = open(self.filelist, 'r')
            filenames = [fs.rstrip() for fs in filelist.readlines()]
            for fs in filenames:
                if not os.path.isfile(self.rootdir + '/' + fs):
                    filenames.remove(fs)
                    print "Cannot find " + fs + " in " + self.rootdir

        if not self.filesuffix is None:
            filenames = fnmatch.filter(filenames, "*" + self.filesuffix + "*")
        filenames.sort()


        self.length = len(filenames)
        if self.verbose:
            print "Found %s nii.gz files..."%self.length
            print "Start Loading"

        self._itemindexes = [0] # [image index of start slice]
        for i, f in enumerate(tqdm(filenames, disable=not self.verbose)) \
                if not self._debug else enumerate(tqdm(filenames[:3], disable=not self.verbose)):
            if self.verbose:
                tqdm.write("Reading from "+f)
            im = sitk.ReadImage(self.rootdir + "/" + f)
            self.dataSourcePath.append(self.rootdir + "/" + f)
            imarr = sitk.GetArrayFromImage(im).astype(self.dtype)
            imarr[imarr <= -1000] = -1000
            self.data.append(from_numpy(imarr))
            self._itemindexes.append(self.data[i].size()[0])
            metadata = {}
            for key in im.GetMetaDataKeys():
                try:
                    if key.split('['):
                        key_type = key.split('[')[0]

                    try:
                        t = NIFTI_DICT[key_type]
                    except:
                        continue
                    metadata[key] = t(im.GetMetaData(key))
                except:
                    metadata[key] = im.GetMetaData(key)
            self.metadata.append(metadata)

        if self._byslices >= 0:
            try:
                self._itemindexes = np.cumsum(self._itemindexes)
                self.length = np.sum([m.size()[self._byslices] for m in self.data])
                self.data = cat(self.data, dim=self._byslices)
            except IndexError:
                print "Wrong Index is used!"
                self.length = len(self.dataSourcePath)

    def ReadRAMRIS(self, csvdata):
        df = pd.read_csv(csvdata)

        ramrisinfo = pd.DataFrame(columns=df.keys())
        row_indexes = []

        pass

    def size(self, int):
        return self.length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.useCatagory:
            index = self._itemindexes[item]
            return self.data[index[0]][index[2]-1]
        elif self.loadCatagory:
            return self.data[item], self._catagories[item]
        else:
            return self.data[item]

    def __str__(self):
        from pandas import DataFrame as df
        s = "==========================================================================================\n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Load By Slice: %i \n" \
            "Image Details:\n" \
            "--------------\n"%(self.rootdir, self.length, self._byslices)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        printable = {'File Name': [], 'Size': [], 'Spacing': [], 'Origin': []}
        for i in xrange(len(self.dataSourcePath)):
            printable['File Name'].append(os.path.basename(self.dataSourcePath[i]))
            # for keys in self.metadata[i]:
            #     if not printable.has_key(keys):
            #         printable[keys] = []
            #
            #     printable[keys].append(self.metadata[i][keys])
            printable['Size'].append([self.metadata[i]['dim[1]'],
                                      self.metadata[i]['dim[2]'],
                                      self.metadata[i]['dim[3]']])
            printable['Spacing'].append([round(self.metadata[i]['pixdim[1]'], 2),
                                         round(self.metadata[i]['pixdim[2]'], 2),
                                         round(self.metadata[i]['pixdim[3]'], 2)])
            printable['Origin'].append([round(self.metadata[i]['qoffset_x'], 2),
                                        round(self.metadata[i]['qoffset_y'], 2),
                                        round(self.metadata[i]['qoffset_z'], 2)])
        data = df(data=printable)
        s += data.to_string()
        return s

    @staticmethod
    def WrapImageWithMetaData(inImage, metadata):
        """WrapImageWithMetaData(np.ndarray or sitk.sitkImage) -> sitk.sitkImage

        :param np.ndarray inImage:
        :return:
        """

        im = inImage
        if isinstance(inImage, np.ndarray):
            im = sitk.GetImageFromArray(im)

        spacing = np.array([metadata['pixdim[1]'], metadata['pixdim[2]'], metadata['pixdim[3]']],
                           dtype=float)
        ori = np.array([-metadata['qoffset_x'], -metadata['qoffset_y'], metadata['qoffset_z']],
                       dtype=float)
        b = float(metadata['quatern_b'])
        c = float(metadata['quatern_c'])
        d = float(metadata['quatern_d'])
        a = np.sqrt(np.abs(1 - b**2 - c**2 - d**2))
        A = np.array([
                [a*a + b*b - c*c - d*d, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
                [2*b*c + 2*a*d , a*a+c*c-b*b-d*d, 2*c*d - 2*a*b],
                [2*b*d - 2*a*c, 2*c*d + 2*a*b, a*a + d*d - c*c - b*b]
            ])
        A[:2, :3] = -A[:2, :3]
        im.SetOrigin(ori)
        im.SetDirection(A.flatten())
        im.SetSpacing(spacing)
        return im

    @staticmethod
    def ResizeToSquare(im, s):
        # Get data
        return resize(im, [s, s], preserve_range=True, mode='constant')


class MaskedTensorDataset(Dataset):
    """
    Data set wrapping like Tensor Dataset, except this also accept a mask.
    """

    def __init__(self, data_tensor, target_tensor, mask_tensor):
        """

        :param ImageDataSet data_tensor:
        :param ImageDataSet target_tensor:
        :param ImageDataSet mask_tensor:
        """
        assert data_tensor.size(0) == target_tensor.size(0) == mask_tensor.size(0)
        assert mask_tensor.dtype == np.uint8, "Mask has to be of dtype np.uint8"

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.mask_tensor = mask_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.mask_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


