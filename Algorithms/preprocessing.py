import SimpleITK as sitk
import numpy as np
import os
from MedImgDataset import ImageDataSet2D
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image, ImageOps

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


def ResizeImagesWithLandmarks(size, root_dir, outputdir, landmarks_csv=None):
    import pandas as pd
    import fnmatch
    from skimage.transform import resize

    assert os.path.isdir(root_dir) and os.path.isdir(outputdir)
    if not landmarks_csv is None:
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
            imsave(outputdir + "/" + row['File'].replace('.jpg', '.png'), newim )
        data = pd.DataFrame.from_dict(outdict)
        data = data[['File', 'Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx']]
        data.to_csv(outputdir + "/Landmarks.csv", index=False)
    else:
        filenames = os.listdir(root_dir)
        filenames.sort()
        dataSourcePath = []
        [dataSourcePath.extend(fnmatch.filter(filenames, "*" + ext)) for ext in ['.png','.jpg']]

        for f in dataSourcePath:
            im = imread(root_dir + "/" + f)
            newim = resize(im, size)
            imsave(outputdir + "/" + f.replace('.jpg', '.png'), newim)
    pass


def PadToSquare(img, s=None):
    """
    Description
    -----------
       Resize image into a square

    :param np.ndarray img: Image that requires resize
    :param int s: specific side
    :return:
    """

    if img.shape[-1] == img.shape[-2]:
        if s == None:
            return img

        from skimage.transform import resize
        outshape = list(img.shape)
        outshape[-2:] = s
        return resize(img, outshape)

    else:

        pilim = Image.fromarray(img)
        pilim.thumbnail(tuple(s))
        background = Image.new('RGB', tuple(s), (0, 0, 0))
        background.paste(
            pilim, (int((s[0] - pilim.size[0]) / 2), int((s[1] - pilim.size[1]) / 2))
        )
        return np.array(background)


def ResizeToSquare(size, root_dir, outputdir, landmarks_csv=None):
    """
    Description
    -----------
      Resize image into a square while preserving the aspect ratio. Transform the landmarks then.
    :param size:
    :param root_dir:
    :param outputdir:
    :param landmarks_csv:
    :return:
    """
    import pandas as pd
    # from scipy.ndimage import imread
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    if not landmarks_csv is None:
        outdict = {'File': [], 'Proximal Phalanx': [], 'Sesamoid': [], 'Metacarpal': [], 'Distal Phalanx': []}
        d = pd.read_csv(landmarks_csv)
        for i, row in d.iterrows():
            im = imread(root_dir + "/" + row['File'].replace("png", 'jpg'), as_grey=False)

            newim = PadToSquare(im, size)

            r, g, b, y = [eval(row[keys]) for keys in ['Proximal Phalanx',
                                                       'Sesamoid',
                                                       'Metacarpal',
                                                       'Distal Phalanx']]
            if im.shape[0] > im.shape[1]:
                r, g, b, y = [[ T[0] * float(size[0]) / im.shape[0] ,
                               (T[1] + float(im.shape[0] - im.shape[1])/2.) * size[1] / im.shape[0] ]
                              for T in [r, g, b, y]]
            elif im.shape[0] < im.shape[1]:
                r, g, b, y = [[(T[0] + float(im.shape[1] - im.shape[0])/2.) * size[0] / im.shape[1] ,
                                T[1] * float(size[1]) / im.shape[1] ]
                              for T in [r, g, b, y]]

            else:
                xfact = im.shape[0] / float(size[0])
                yfact = im.shape[1] / float(size[1])
                r, g, b, y = [[int(T[0] / xfact), int(T[1] / yfact)] for T in [r, g, b, y]]

            r, g, b, y = [[int(T[0]), int(T[1])] for T in [r, g, b, y]]

            outdict['File'].append(os.path.basename(row['File'].replace('jpg', 'png')))
            outdict['Proximal Phalanx'].append(r)
            outdict['Sesamoid'].append(g)
            outdict['Metacarpal'].append(b)
            outdict['Distal Phalanx'].append(y)
            imsave(outputdir + "/" + row['File'].replace('.jpg', '.png'), newim )


        data = pd.DataFrame.from_dict(outdict)
        data = data[['File', 'Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx']]
        data.to_csv(outputdir + "/Landmarks.csv", index=False)
        PlotImageWithLandmarks("./TOCI/05.Resized_SAR")
    else:
        images = ImageDataSet2D(root_dir, dtype=np.uint8, verbose=True, as_gray=False)
        imdirs = images.dataSourcePath
        for i in xrange(len(images)):
            newim = PadToSquare(images[i].numpy(), size)
            imsave(imdirs[i].replace(root_dir, outputdir).replace(".jpg", '.png'), newim)



def ReadFeatures(landmarks_csv):
    """ReadFeatures(landmarks_csv)->np.ndarray
    Description
    -----------
      Read the data into a nparray

    :param landmarks_csv:
    :return:
    """
    assert os.path.isfile(landmarks_csv)
    import pandas as pd

    out = []
    d = pd.read_csv(landmarks_csv)
    for i, row in d.iterrows():
        r, g, b, y = [eval(row[keys]) for keys in ['Proximal Phalanx',
                                                   'Sesamoid',
                                                   'Metacarpal',
                                                   'Distal Phalanx']]
        r, g, b, y = [np.expand_dims(T, 0) for T in [r, g, b, y]]
        t = np.concatenate([r, g, b, y], axis=0)
        t = np.expand_dims(t, 0)
        out.append(t)
    out = np.concatenate(out, 0)
    return out


def SaveFeatures(imfname, features, columnName=('Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx')):
    """SaveFeatures -> dict
    Description
    -----------
      Save the features according to the input tensor. Assume the tensor is a NxMx2 array where
      N is should equal the number of images and M equal number or features per

    :param iter         imfname:    Filename list that corresponds to the images with features
    :param np.ndarray   features:   Tensor of NxMxN dimensions
    :param iter         columnName: Name of the output dictionary columns
    :return: dict
    """
    assert len(imfname) == features.shape[0]
    assert len(columnName) == features.shape[1]
    outdict = {}
    for keys in ['File']+list(columnName):
        outdict[keys] = []

    for i in xrange(len(imfname)):
        outdict['File'].append(imfname[i])
        for j, keys in enumerate(list(columnName)):
            outdict[keys].append(features[i, j].tolist())
    return outdict


def DataAugmentatation(root_dir, outputdir, landmarks_csv):
    import pandas as pd
    from visualization import VisualizeMapWithLandmarks
    assert os.path.isdir(root_dir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    seg = iaa.Sequential([iaa.Fliplr(0.7), iaa.Affine(rotate=(-70, 70), order=[0, 1])])

    data = ImageDataSet2D(root_dir, dtype=np.float32, verbose=True, as_grey=True)
    paths = data.dataSourcePath
    data = data.tonumpy()

    keypoints = []
    features = ReadFeatures(landmarks_csv)
    for i in xrange(features.shape[0]):
        pts = []
        for j in xrange(len(features[i])):
            y, x = features[i, j]
            pts.append(ia.Keypoint(x=x, y=y))
        keypoints.append(ia.KeypointsOnImage(pts, shape=tuple(data[i].shape)))

    # VisualizeMapWithLandmarks(data, features, env="Test", N=100, win="before")
    seg_det = seg.to_deterministic()

    images_aug = seg_det.augment_images(data)
    keypoints_aug = seg_det.augment_keypoints(keypoints)
    images_aug = np.concatenate([np.expand_dims(I, 0) for I in images_aug], 0)
    keypoints_aug = np.stack([[[p.y, p.x] for p in keypoints_aug[i].keypoints] for i in xrange(len(keypoints_aug))])

    # VisualizeMapWithLandmarks(images_aug, keypoints_aug, env="Test", N=100, win="after")
    outfnames = [os.path.basename(d.replace(".jpg", ".png").replace(".png", "_AUG.png")) for d in paths]
    for i, im in enumerate(images_aug):
        im = np.tile(np.array(im*255, dtype=np.uint8)[:,:,None], (1, 1, 3))
        imsave(outputdir + "/" + outfnames[i], im)
    f = pd.DataFrame.from_dict(SaveFeatures(outfnames, keypoints_aug))
    f = f[['File', 'Proximal Phalanx', 'Sesamoid', 'Metacarpal', 'Distal Phalanx']]
    f.to_csv(outputdir + "/Landmarks.csv", index=False)


def CropThumb(im, features, patchsize):
    """
    Description
    -----------
      Crop thumb from image. Ignore second row if features is 4x2 array

    :param np.ndarray im:
    :param np.ndarray features: 4x2 array or 3x2 array
    :return:
    """
    if features.shape[0] == 4:
        r, g, b = [features[0], features[2], features[3]]
    else:
        r, g, b = features
    r, g, b = [np.array(x) for x in [r, g, b]]

    # cent = np.array([r + g + b]) /3.
    # cent = np.array(cent, dtype=int)
    vect = b - g
    vect = vect / np.linalg.norm(np.array(vect, dtype=float))
    deg  = np.arccos(-vect[0])
    deg = np.rad2deg(deg)
    print deg

    halfsize = np.array(im.shape[:2]) / 2
    bounds = (halfsize[1] - patchsize/2, halfsize[0] - patchsize/2, halfsize[1] - patchsize/2, halfsize[0] - patchsize/2) # top, right, bottom, left
    # seg = [iaa.Sequential([iaa.Affine(translate_px={'x':  - c[1] + halfsize[0], 'y': - c[0] + halfsize[1]}),
    #
    seg = [iaa.Sequential([iaa.Affine(translate_px={'x':  - c[1] + halfsize[0], 'y': - c[0] + halfsize[1]}),
                           iaa.Affine(rotate=-deg),
                           iaa.Crop(px=bounds, keep_size=False)]) for c in [r, b]]
    images = [S.augment_image(im) for S in seg]
    return images

def ExtractROIs(root_dir, landmark_csv, outdir):
    """

    :param root_dir:
    :param landmark_csv:
    :param outdir:
    :return:
    """
    assert os.path.isdir(root_dir)
    assert os.path.isfile(landmark_csv)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    features = ReadFeatures(landmark_csv)
    images = ImageDataSet2D(root_dir,dtype=np.uint8, verbose=True, as_grey=False)
    paths = images.dataSourcePath
    images = images.tonumpy()

    assert len(images) == features.shape[0]
    for i, row in enumerate(zip(images, features)):
        ims = CropThumb(row[0][:,:,0], row[1], 64)
        ext = os.path.basename(paths[i]).split('.')[-1]
        outnames = paths[i].replace(root_dir, outdir).replace('.' + ext, '_ROIs.' + ext)
        outim = np.zeros(shape=[64, 64, 3])
        outim[:,:,0] = ims[0]
        outim[:,:,1] = ims[1]
        imsave(outnames, outim)


if __name__ == '__main__':
    # PlotImageWithLandmarks("./TOCI/04.Resized")
    # ExtractLandmarks("./TOCI/02.ALL")
    # ResizeImagesWithLandmarks([512,512], "./TOCI/02.ALL", "./TOCI/03.Annotated/Landmarks.csv", "./TOCI/04.Resized/")
    # ResizeToSquare([512, 512], "./TOCI/10.TestData","./TOCI/10.TestData/Resized_SAR")
    # ResizeToSquare([512, 512], "./TOCI/02.ALL","./TOCI/05.Resized_SAR", landmarks_csv="./TOCI/03.Annotated/Landmarks.csv")
    # PlotImageWithLandmarks("./TOCI/05.Resized_SAR")
    # DataAugmentatation("./TOCI/05.Resized_SAR", "./TOCI/05.Resized_SAR/aug", "./TOCI/05.Resized_SAR/Landmarks.csv")
    # ReadFeatures("./TOCI/03.Annotated/Landmarks.csv")
    ExtractROIs("./TOCI/05.Resized_SAR", "./TOCI/05.Resized_SAR/Landmarks.csv", "./TOCI/05.Resized_SAR/ROIs")
    pass