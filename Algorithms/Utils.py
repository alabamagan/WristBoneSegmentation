import os
import visdom
import fnmatch
import numpy as np
import sys
import os
vis = visdom.Visdom(server="http://137.189.141.212", port=80)

__all__ = ["ShowImages", "CreateTrainingSamples", "RemoveSpeicfiedSlices"]

def MoveFiles(prefix, fn, postix):
    print "Moving %s to %s..."%(fn, postix)
    os.system("mv " + prefix + "/" + fn + "* " + prefix + "/" + postix)
    return

def ShowImages(dir, pattern=None):
    """
    Description
    -----------
      Show all .npy files in a directory with visdom

    :param args:
    :return:
    """

    disrange = [-1000, 400]

    files = os.listdir(dir)

    if not(pattern is None):
        files = fnmatch.filter(files, pattern)
    files.sort()

    for j in xrange(files.__len__()):
        s = np.load(dir + "/" + files[j])
        s = np.array(s, dtype=float)
        s = np.clip(s, disrange[0], disrange[1])
        s -= s.min()
        s /= s.max()
        vis.text(files[j], win="ShowImages_Filename")
        vis.image(s, win="ShowImages")

    pass



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