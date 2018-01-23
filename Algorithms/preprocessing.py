import SimpleITK as sitk
import numpy as np
import os
import ntpath, fnmatch
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


def ResampleImage(target, ref, output_prefix, upscalefactor=2):
    """
      Resample the target image while maintaining its inter slice spacing

    :param target:
    :param ref:
    :param output_prefix:
    :return:
    """

    assert len(target) == len(ref), "Different number of files specified"

    for i in xrange(len(target)):
        print i
        assert os.path.isfile(target[i]) and os.path.isfile(ref[i])

        imtarget = sitk.ReadImage(target[i])
        imref = sitk.ReadImage(ref[i])

        # Recalculate spacing and size
        outputspacing = np.array(imref.GetSpacing())
        outputsize = np.array(imref.GetSize())
        L = outputspacing*outputsize
        for j in xrange(upscalefactor):
            outputsize[2] += outputsize[2] - 1
        outputspacing = L / outputsize

        filter = sitk.ResampleImageFilter()
        filter.SetOutputDirection(imref.GetDirection())
        filter.SetOutputOrigin(imref.GetOrigin())
        filter.SetOutputSpacing(outputspacing)
        filter.SetSize(outputsize)
        saveim = filter.Execute(imtarget)
        outname = target[i].replace(ntpath.basename(target[i]), output_prefix + ntpath.basename(target[i]))
        sitk.WriteImage(saveim, outname)



# def main():
#     dirs = RecursiveListDir(2, "./SBL_2D_3D")
#
#     for d in dirs:
#         reader = sitk.ImageSeriesReader()
#         if len(reader.GetGDCMSeriesIDs(d)) != 0:
#             outname = "./SBL_2D_3D" + d.replace("./SBL_2D_3D", "").replace('/', '_') + ".nii.gz"
#             reader.SetFileNames(reader.GetGDCMSeriesFileNames(d))
#             sitk.WriteImage(reader.Execute(), outname, True)


"""
Recursively resample the images
"""
# def main():
#     dir = "./SBL_2D_3D/01.NIIDIR/"
#     f = os.listdir(dir)
#     ftarget = fnmatch.filter(f, "*T1_3D*")
#     fref = fnmatch.filter(f, "*T1_COR*")
#     ftarget = [dir + F for F in ftarget]
#     fref = [dir + F for F in fref]
#     ftarget.sort()
#     fref.sort()
#     # print ftarget
#     # print fref
#     ResampleImage(ftarget, fref, "RES-", 2)

def main():
    dir = "../SBL_2D_3D/10.3D_BrainMask"
    refdir = "../SBL_2D_3D/13.3D_Resampled"
    outdir = "../SBL_2D_3D/14.BrainMask_Resampled"

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    mfiles = os.listdir(dir)
    mfiles.sort()

    reffiles = os.listdir(refdir)
    reffiles.sort()

    for index, tar in enumerate(mfiles):
        im = sitk.ReadImage(dir + "/" + tar)
        ref = sitk.ReadImage(refdir + "/" + reffiles[index])
        outname = outdir + "/RES-" + tar
        F = sitk.ResampleImageFilter()
        F.SetReferenceImage(ref)
        outim = F.Execute(im)
        sitk.WriteImage(outim, outname)



if __name__ == '__main__':
    main()
