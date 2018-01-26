import SimpleITK as sitk
import numpy as np
import os
from MedImgDataset import ImageDataSet2D
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
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


def ExtractLandmarks(rootdir):
    from pandas import DataFrame as df
    im = ImageDataSet2D(rootdir, readmode='RGB', dtype=np.uint8, verbose=True)
    out = {'File': [], 'Proximal Phalanx': [], 'Sesamoid': [], 'Metacarpal': [], 'Distal Phalanx': []}
    for i, sample in enumerate(im):
        try:
            temp = sample.numpy()
            coordblue = np.where(np.all(temp == (0, 0, 255), axis=-1))
            coordred = np.where(np.all(temp == (255, 0, 0), axis=-1))
            coordyellow = np.where(np.all(temp == (255, 255, 0), axis=-1))
            coordgreen = np.where(np.all(temp == (0, 255, 0), axis=-1))
            coordblue, coordred, coordyellow, coordgreen \
                = [zip(z[0], z[1]) for z in [coordblue, coordred, coordyellow, coordgreen]]
            coordblue, coordred, coordyellow, coordgreen \
                = [z[0] if len(z) > 0 else z  for z in [coordblue, coordred, coordyellow, coordgreen]]
            out['File'].append(os.path.basename(im.dataSourcePath[i]))
            out['Proximal Phalanx'].append(coordred)
            out['Sesamoid'].append(coordgreen)
            out['Metacarpal'].append(coordblue)
            out['Distal Phalanx'].append(coordyellow)
        except:
            continue

    data = df(data=out)
    data = data[['File', 'Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx']]
    data.to_csv(rootdir + "/Landmarks.csv", index=False)
    print data.to_string()


def PlotImageWithLandmarks(rootdir):
    import pandas as pd

    landmarkds = pd.read_csv(rootdir + "/Landmarks.csv")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, row in landmarkds.iterrows():
        print i
        ax1.cla()
        ax1.imshow(imread(rootdir + "/" + row['File']))
        coords = np.array([eval(row[keys]) for keys in ['Metacarpal', 'Proximal Phalanx', 'Distal Phalanx']])
        coords = coords.T
        ax1.plot(coords[1], coords[0], linestyle='-', linewidth=2)
        plt.draw()
        plt.pause(0.2)


def ResizeImagesWithLandmarks(size, root_dir, landmarks_csv, outputdir):
    import pandas as pd
    from skimage.transform import resize

    assert os.path.isdir(root_dir) and os.path.isdir(outputdir)
    assert os.path.isfile(landmarks_csv)

    outdict = {'File': [], 'Proximal Phalanx': [], 'Sesamoid': [], 'Metacarpal': [], 'Distal Phalanx': []}
    d = pd.read_csv(landmarks_csv)
    for i, row in d.iterrows():
        im = imread(root_dir + "/" + row['File'].replace("png", 'jpg'))

        newim = resize(im, size)

        r, g, b, y = [eval(row[keys]) for keys in ['Proximal Phalanx',
                                                   'Sesamoid',
                                                   'Metacarpal',
                                                   'Distal Phalanx']]

        xfact = im.shape[0] / float(size[0])
        yfact = im.shape[1] / float(size[1])

        r, g, b, y = [[int(T[0] / xfact), int(T[1] / yfact)] for T in [r, g, b, y]]
        outdict['File'].append(os.path.basename(row['File'].replace('jpg', 'png')))
        outdict['Proximal Phalanx'].append(r)
        outdict['Sesamoid'].append(g)
        outdict['Metacarpal'].append(b)
        outdict['Distal Phalanx'].append(y)
        imsave(outputdir + "/" + row['File'].replace('.jpg', 'png'), newim )
    data = pd.DataFrame.from_dict(outdict)
    data = data[['File', 'Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx']]
    data.to_csv(outputdir + "/Landmarks.csv", index=False)
    pass

if __name__ == '__main__':
    PlotImageWithLandmarks("./TOCI/04.Resized")
    # ExtractLandmarks("./TOCI/02.ALL")
    # ResizeImagesWithLandmarks([512,512], "./TOCI/02.ALL", "./TOCI/03.Annotated/Landmarks.csv", "./TOCI/04.Resized/")