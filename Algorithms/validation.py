from MedImgDataset import ImageDataSet
from torch.utils.data import TensorDataset
import numpy as np
from scipy.spatial.distance import dice

def DICE(tensordata):
    """DICE(TensorDataset)
    Description
    -----------
      Calculate the dice information of the tensor dataset with first column equals
      segmentation and second columen equals groundtruth

    :param tensordata:
    :return:
    """

    seg, gt = [], []
    for i, data in enumerate(tensordata):
        seg.extend(data[0].numpy().flatten().tolist())
        gt.extend(data[1].numpy().flatten().tolist())
        # d1 = data[0].numpy().flatten()
        # d2 = data[1].numpy().flatten()
        # print dice(d1, d2)
    # pass
    seg = np.array(seg)
    gt = np.array(gt)
    return 1-dice(seg, gt)


if __name__ == '__main__':
    seg = ImageDataSet("../ERA_Segmentation/03_TEST/output", dtype=np.uint8, verbose=True)
    gt = ImageDataSet("../ERA_Segmentation/03_TEST/gt", dtype=np.uint8, verbose=True)
    tensor = TensorDataset(seg, gt)
    print DICE(tensor)