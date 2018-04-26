import SimpleITK as sitk
import numpy as np
import fnmatch
import os
from MedImgDataset import ImageData
sitk.ProcessObject_GlobalWarningDisplayOff()


def RecursiveListDir(searchDepth, rootdir):
    """
      Recursively lo
    :param searchDepth:
    :param rootdir:
    :return:
    """
    dirs = os.listdir(rootdir)
    nextlayer = []
    for D in dirs:
        if os.path.isdir(rootdir + "/" + D):
            nextlayer.append(rootdir + "/" + D)
    DD = []
    if searchDepth >= 0 and len(nextlayer) != 0:
        for N in nextlayer:
            K = RecursiveListDir(searchDepth - 1, N)
            if not K is None:
                DD.extend(K)
    DD.extend(nextlayer)
    return DD


def MoveFiles(prefix, fn, postix):
    print "Moving %s to %s..."%(fn, postix)
    os.system("mv " + prefix + "/" + fn + "* " + prefix + "/" + postix)
    return


def CreateTrainingSamples(num, dir):
    """
    Description
    ------------
      Draw samples from a set of directories

    :param int num:
    :param str dir:
    :return:
    """

    import fnmatch
    import random
    import multiprocessing as mpi

    assert isinstance(dir, str), "Directory must be a string!"
    assert isinstance(num, int), "Num must be integer!"
    assert os.path.isdir(dir), "Directory doesn't exist!"

    dirfiles = os.listdir(dir)
    npfiles = fnmatch.filter(dirfiles, "*.npy")

    # Count number of samples in folder
    uniqueSamples = [fs.split('_')[0] for fs in npfiles]
    uniqueSamples = list(set(uniqueSamples))

    assert len(uniqueSamples) > num, "Required unique samples greater than" \
                                     "total number of samples in the directory"

    # Create output directory if not exist
    if not os.path.isdir(dir + "/train"):
        os.mkdir(dir + "/train")
    if not os.path.isdir(dir + "/test"):
        os.mkdir(dir + "/test")

    # Choose from original directory
    trainsamples = random.sample(np.arange(len(uniqueSamples)), num)
    trainsamples = [uniqueSamples[i] for i in trainsamples]

    pool = mpi.Pool(processes=8)
    p = []

    # Move files
    for fs in uniqueSamples:
        if fs in trainsamples:
            process = pool.apply_async(MoveFiles, args=[dir, fs, "train"])
            p.append(process)
            # MoveFiles(fs, "train")
        else:
            process = pool.apply_async(MoveFiles, args=[dir, fs, "test"])
            p.append(process)
            # MoveFiles(fs, "test")

    # Wait till job finish
    for process in p:
        process.wait()

    pass

def RemoveSpeicfiedSlices(dir, spec):
    """
    Description
    -----------
      Clean directory and only keep slices and files specified by param spec.
      Spec should either be a directory or a nested list.

      The file should be arranged in the following fashion using space as separator:
      #1  [unique_sample_prefix_1] [start slice_1] [end slice_1]
      #2  [unique_sample_prefix_2] [start slice_2] [end slice_2]
      #3  ...

      The nested list should be arranged in the following fashion:
      [ [unique_sample_prefix_1, start_slice_1, end_slice_1],
        [unique_sample_prefix_2, start_slice_2, end_slice_2],
        ... ]

    :param str dir:
    :param str/list spec:
    :return:
    """

    import re
    import multiprocessing as mpi

    # Create directory to hold removed slices
    if not os.path.isdir(dir + "/removed"):
        os.mkdir(dir + "/removed")

    # Read file into a list
    if isinstance(spec, str):
        temp = []
        for line in file(spec, 'r').readlines():
            temp.append(line.replace('\n', '').split(' '))
        spec = temp

    files = os.listdir(dir)
    uniquesamples = list(set([fs.split('_')[0] for fs in files]))
    specSamples = [s[0] for s in spec]

    # Check if any samples are not in the files
    if len(files) != len(uniquesamples):
        for us in uniquesamples:
            if not (us in specSamples):
                os.system("mv " + us + "* " + dir + "/removed")

    # Identify which files to be moved
    tobemoved = []
    for i in xrange(len(spec)):
        fs = fnmatch.filter(files, spec[i][0] + "*")
        for ff in fs:
            result = re.match(r'.*S([0-0]+).*', ff)
            if result is None:
                print "Pattern error for file: " + ff
                continue

            sliceNum = int(result.group(1))
            if sliceNum < spec[i][1] or sliceNum >= spec[i][2]:
                tobemoved.append(dir + "/" + ff)

    # Move files
    indexlist = range(len(tobemoved))
    indexlist = indexlist[::10000]
    if indexlist[-1] != len(tobemoved) - 1:
        indexlist.append(len(tobemoved) - 1)

    pool = mpi.Pool(processes=6)
    p = []
    for i in xrange(len(indexlist) - 1):
        arg = " ".join(tobemoved[indexlist[i]:indexlist[i+1]])
        com = "mv " + arg + " " + dir + "/removed"
        p.append(pool.apply_async(os.system, arg=[com]))

    for process in p:
        process.wait()

    pass


def CheckDir(dir):
    """
    Description
    -----------
      Check if directory has the correct format

    :param str dir:
    :return:
    """

    assert os.path.isdir(dir), "Directory doesn't exist!"

    files = os.listdir(dir)
    files = fnmatch.filter(files, "*.npy")
    files.sort()

    uniquesamples = list(set([ff.split('_')[0] for ff in files]))
    suffix = list(set([ff.split('_')[1] for ff in files]))
    suffix.sort()
    print suffix

    for f in uniquesamples:
        slices = []
        for suf in suffix:
            fs = fnmatch.filter(files, f + "_" + suf + "_*")
            slices.append(str(len(fs)))
        print f + ": " + ",".join(slices)
    print "Total: ", len(uniquesamples)


def Normalize(array, lowerbound=None, upperbound=None, range=[0, 1]):
    """
    Description
    -----------
      Clip the image with lowerbound and upperbound and then normalize it to the given range.

    :param np.ndarray array:
    :param float lowerbound:
    :param float upperbound:
    :param list range:
    :return: np.ndarray
    """

    assert isinstance(array, np.ndarray)
    assert isinstance(range, list)

    t = np.array(array, dtype=float)
    if not (lowerbound is None or upperbound is None):
        assert lowerbound < upperbound, "Lower bound must be less than upper bound!"
        t = np.clip(t, lowerbound, upperbound)
        t = t - lowerbound
    else:
        t = array
        t = t - t.min()

    if not (upperbound is None):
        t = t * (range[1] - range[0]) / (upperbound -lowerbound) - range[0]
    else:
        t = t * (range[1] - range[0]) / t.max() - range[0]

    return t


def RotateImage(im):
    """
    Description
    -----------
      Rotate input image by 180 degrees

    :param np.ndarray/str im:
    :return:
    """
    if isinstance(im, np.ndarray):
        return  np.rot90(np.rot90(im))
    else:
        assert os.path.isfile(im), "Cannot find file!"

        image = np.load(im)
        assert image.ndim == 2, "Cannot rotate 3D image!"

        np.save(im, np.rot90(np.rot90(image)))

def BatchRotateImage(dir, exception=[]):
    """
    Description
    -----------
      Rotate all the numpy images by 180 degrees clockwise if the image is 2D. Overwrites the image.

    :param str dir:
    :param list exception:
    :return:
    """

    import multiprocessing as mpi

    assert os.path.isdir(dir), "Directory doesn't exist!"
    assert isinstance(exception, list), "Exception must be in form of a list!"

    pool = mpi.Pool(processes=8)
    process = []

    fs = os.listdir(dir)
    fs = fnmatch.filter(fs, "*.npy")
    fs = [ff for ff in fs if not ff in exception]
    fs.sort()

    for ff in fs:
        print "Working on ", ff
        im = dir + "/" + ff
        p = pool.apply_async(RotateImage, args=[im])
        process.append(p)

    for i in xrange(len(process)):
        process[i].wait()
        print "Progress: ", i * 100 / float(len(process)), "%"

    pass


def MergeLabels(label1, label2, outname):
    """
    Description
    -----------
      Merge two label into one binary label. Must be nii format.

    :param str label1:
    :param str label2:
    :param str outname:
    :return:
    """

    assert os.path.isfile(label1) and os.path.isfile(label2)


    l1 = sitk.Cast(sitk.ReadImage(label1), sitk.sitkUInt8)
    l2 = sitk.Cast(sitk.ReadImage(label2), sitk.sitkUInt8)
    L = sitk.Add(l1, l2)
    L = sitk.BinaryThreshold(L, 1, 255, 1, 0)

    sitk.WriteImage(L, outname)
    pass


def CopyRecusrsivelyForKeyWards():
    rootdir = "../ERA_Segmentation/01_RAW"
    keywords = "segmentation_carpal"
    prefix = ""
    suffix = "_A_Carpal"
    outdir = "11_CARPALS"
    dirs = RecursiveListDir(3, rootdir)
    dirs.sort()
    for i, d in enumerate(dirs):
        files = os.listdir(d)
        for f in files:
            if f.find(keywords) > 0:
                # print i, d, f
                os.system("cp %s %s"%(d + "/" + f, rootdir.replace("01_RAW", outdir + "/") + d.replace(rootdir+"/", "").replace("/ROI", "") + suffix + ".nii"))


def RenameIndexes(dir):
    """
    Description
    -----------
      Rename the files with numeric indexes before charactor '_' to '%03d' format

    :param dir:
    :return:
    """

    files = os.listdir(dir)
    for f in files:
        x = int(f.split('_')[0])
        F = f.replace(f.split('_')[0], "%03d"%x)
        os.rename(dir + "/" + f, dir + "/" + F)
    pass

if __name__ == '__main__':
    from torch.utils.data import dataloader
    r1 = "../ERA_Segmentation/11_CARPALS"
    r2 = "../ERA_Segmentation/12_META_CARPALS"

    d1 = ImageData.ImageDataSet(r1, dtype=np.uint8, verbose=True)
    d2 = ImageData.ImageDataSet(r2, dtype=np.uint8, verbose=True)

    print d1, d2

    for i, P in enumerate(zip(d1.dataSourcePath, d2.dataSourcePath)):
        try:
            MergeLabels(P[0], P[1], d1.dataSourcePath[i].replace('_A_', '_M_').replace("11_CARPALS", "23_MERGED_ONELABEL"))
        except:
            print i, " has some problem"

