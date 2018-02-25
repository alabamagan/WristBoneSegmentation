import visdom
import numpy as np
from torch.tensor import _TensorBase

vis = visdom.Visdom(port=80)

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



def Visualize3D(*args, **kwargs):
    """Visaulize3D(*args, **kwargs) -> None

    Keywords:
    :keyword int axis   Which axis to display
    :keyword str env    Visdom environment
    :keyword str prefix Window prefix of visdom display windows
    :keyword int nrow   Number of images per row
    :keyword iter displayrange  Display values range
    :keyword iter indexrange    Index range of the displayed image

    :return: None
    """
    raise NotImplementedError


def Visualize2D(*args, **kwargs):
    """Visualize2D(*args, **kwargs) -> None

    Keywords:
    :keyword int axis   Which axis to display
    :keyword str env    Visdom environment
    :keyword str prefix Window prefix of visdom display windows
    :keyword int nrow   Number of images per row
    :keyword iter displayrange  Display values range
    :keyword iter indexrange    Index range of the displayed image

    :return: None
    """
    axis = kwargs['axis'] if kwargs.has_key('axis') else 0
    env = kwargs['env'] if kwargs.has_key('env') else "Name"
    prefix = kwargs['prefix'] if kwargs.has_key('prefix') else 'Image'
    displayrange = kwargs['displayrange'] if kwargs.has_key('displayrange') else [0, 0]
    indexrange = kwargs['indexrange'] if kwargs.has_key('indexrange') else [0, 15]
    nrow = kwargs['nrow'] if kwargs.has_key('nrow') else 5


    for i, tensor in enumerate(args):
        assert issubclass(type(tensor), _TensorBase)
        t = tensor.permute(*np.roll(range(3), -axis).tolist())
        temp = t.numpy()
        if displayrange == [0, 0]:
            drange = [0, 0]
            drange[0] = temp.min()
            drange[1] = temp.max() + 0.1
        else:
            drange = displayrange
        temp = Normalize(temp, drange[0], drange[1])
        newRange = [max(0, indexrange[0]), min(indexrange[1], temp.shape[0])]

        vis.images(np.expand_dims(temp, 1)[newRange[0]:newRange[1]],
                   nrow=nrow, env=env, win=prefix+"%i"%i)


def VisualizeMapWithLandmarks(images, landmarks, env="TOCI", N=20, win="Image"):
    """
    Description
    -----------
      Visualize images landmarkds
    :param np.ndarray images:
    :param np.ndarray landmarks:
    :return:
    """
    from skimage.draw import circle, line

    color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

    if images.ndim == 3:
        assert images.shape[0] == landmarks.shape[0]
        landmarks = np.array(landmarks, dtype=np.uint32)
        images = np.tile(images[:,None,:,:], [1, 3, 1, 1])
        for i in xrange(0, np.min([N, images.shape[0]])):
            for j in xrange(landmarks.shape[1]):
                # rr, cc = line(landmarks[i,j,0], landmarks[i,j,1], landmarks[i,j-1,0], landmarks[i,j-1,1])
                rr, cc = circle(landmarks[i,j,0], landmarks[i,j,1], 5, shape=images.shape[2:])
                images[i, :, rr, cc] = color[j]
        vis.images(images[:np.min([N, images.shape[0]])], nrow=5, env=env, win=win)
    else:
        for i in xrange(landmarks.shape[0]):
            rr, cc = circle(landmarks[i,0], landmarks[i,1], 3, shape=images.shape)
            images[rr, cc] = 255
        vis.image(images, env=env, win=win)

