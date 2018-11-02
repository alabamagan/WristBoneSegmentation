import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from MedImgDataset import ImageDataSet
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def perf_measure(y_actual, y_guess):
    y = y_actual.flatten()
    x = y_guess.flatten()

    TP = np.sum(np.multiply(y == True, x == True))
    TN = np.sum(np.multiply(y == False, x == False))
    FP = np.sum(np.multiply(y == False, x == True))
    FN = np.sum(np.multiply(y == True, x == False))
    return TP, FP, TN, FN

def JAC(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)
    return TP / (TP + FP + FN)

def GCE(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)
    n = float(np.sum(TP + FP + TN + FN))
    return np.min([FN * (FN + 2*TP) / (TP + FN) + FP * (FP + 2*TN)/(TN+FP),
                FP * (FP + 2*TP) / (TP + FP) + FN * (FN + 2*TN)/(TN+FN)]) / n

def DICE(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)

    return 2*TP / (2*TP+FP+FN)

def VS(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)

    return 1 - abs(FN - FP) / (2*TP + FP + FN)

def EVAL(seg, gt):
    df = pd.DataFrame(columns=['filename','ImageIndex','SLICE', 'GCE', 'JAC', 'DICE', 'VD', 'Catagory'])

    for i, row in enumerate(tqdm(zip(seg, gt))):
        s = row[0]
        g = row[1]
        cat = s[1]
        ss = s[0].numpy().flatten()
        gg = g[0].numpy().flatten()
        if np.sum(gg) == 0 or np.sum(ss) == 0:
            continue
        d = DICE(ss, gg)
        jac = JAC(ss, gg)
        gce = GCE(ss, gg)
        vs = VS(ss, gg)
        imindex = np.argmax(seg._itemindexes > i)
        data = pd.DataFrame([[os.path.basename(seg.dataSourcePath[imindex - 1]), imindex, i, gce, jac, d, 1-vs, cat]],
                            columns=['filename','ImageIndex','SLICE', 'GCE', 'JAC', 'DICE', 'VD','Catagory'])
        df = df.append(data)
    return df


if __name__ == '__main__':
    seg = ImageDataSet("../ERA_Segmentation/03_TEST/postprocessed/", dtype=np.uint8, verbose=True)
    seg_nocat = ImageDataSet("../ERA_Segmentation/03_TEST/postprocessed_nocat/", dtype=np.uint8, verbose=True)
    gt = ImageDataSet("../ERA_Segmentation/03_TEST/gt", dtype=np.uint8, verbose=True)
    seg.LoadWithCatagories("../ERA_Segmentation/CaseSegment.txt")
    seg_nocat.LoadWithCatagories("../ERA_Segmentation/CaseSegment.txt")
    gt.LoadWithCatagories("../ERA_Segmentation/CaseSegment.txt")

    ramris = pd.read_csv("../ERA_Segmentation/03_TEST/list.csv")

    dd = EVAL(seg, gt)
    dd_nocat = EVAL(seg_nocat, gt)
    dd['UseCatagory'] = 1
    dd_nocat['UseCatagory'] = 0
    dd_nocat['Catagory'] = 0
    dd = pd.merge(ramris,dd)
    dd_nocat = pd.merge(ramris,dd_nocat)
    final = dd.append(dd_nocat)
    final.to_csv("../ERA_Segmentation/03_TEST/results.csv")
    print final.to_string()
    print final['DICE'][final['Catagory']==2].mean()

    # dd.to_csv("../ERA_Segmentation/validation.csv")
    # print perf_measure(seg[10].numpy().flatten(), gt[10].numpy().flatten())
    # print D(seg[10].numpy().flatten(), gt[10].numpy().flatten())
    # print 1-dice(seg[10].numpy().flatten(), gt[10].numpy().flatten())

    # target = np.array([1, 1, 0, 0])
    # guess = np.array([0, 1, 0, 0])
    #
    # print perf_measure(target, guess)
    # print 1 - dice(target, guess)
    # print D(target, guess)