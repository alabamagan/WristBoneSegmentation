from torch.utils.data import Dataset
from torch import from_numpy
import fnmatch
import os
import numpy as np
import SimpleITK as sitk

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
    def __init__(self, rootdir, verbose=False, dtype=float):
        """

        :param rootdir:
        """
        super(Dataset, self)
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.metadata = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self._ParseRootDir()

    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all nii images to cache

        :return:
        """

        if self.verbose:
            print "Parsing root path: ", self.rootdir
        filenames = os.listdir(self.rootdir)
        filenames = fnmatch.filter(filenames, "*.nii.gz")
        filenames.sort()


        self.length = len(filenames)
        if self.verbose:
            print "Found %s nii.gz files..."%self.length
            print "Start Loading"

        for f in filenames:
            im = sitk.ReadImage(self.rootdir + "/" + f)
            self.dataSourcePath.append(self.rootdir + "/" + f)
            self.data.append(from_numpy(np.array(sitk.GetArrayFromImage(im), dtype=self.dtype)))
            metadata = {}
            for key in im.GetMetaDataKeys():
                try:
                    if key.split('['):
                        key_type = key.split('[')[0]
                    t = NIFTI_DICT[key_type]
                    metadata[key] = t(im.GetMetaData(key))
                except:
                    metadata[key] = im.GetMetaData(key)
            self.metadata.append(metadata)

    def size(self, int):
        return self.length

    def __len__(self):
        return self.length

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
        printable = {'File Name': [], 'Size': [], 'Spacing': [], 'Origin': []}
        for i in xrange(self.length):
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


