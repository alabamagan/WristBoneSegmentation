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

    TP = np.sum((y == True) & (x == True))
    TN = np.sum((y == False) & (x == False))
    FP = np.sum((y == False) & (x == True))
    FN = np.sum((y == True) & (x == False))
    return TP, FP, TN, FN

def JAC(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)
    return TP / (TP + FP + FN)

def GCE(actual, guess):
    TP, FP, TN, FN = np.array(perf_measure(actual, guess), dtype=float)
    n = float(np.sum(TP + FP + TN + FN))

    val = np.min([FN * (FN + 2*TP) / (TP + FN) + FP * (FP + 2*TN)/(TN+FP),
                FP * (FP + 2*TP) / (TP + FP) + FN * (FN + 2*TN)/(TN+FN)]) / n
    # if np.sum(actual) == 0 or  np.sum(guess) == 0:
    #     print TP, FP, TN, FN, np.sum(actual) == 0, np.sum(guess) == 0
    return val

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
        # if np.sum(gg) == 0 and np.sum(ss) == 0:
        #     if np.sum(ss) == 0 and int(cat.data.numpy()) != 0:
        #         imindex = np.argmax(seg._itemindexes > i)
        #         fname = os.path.basename(seg.dataSourcePath[imindex - 1])
        #         print i, ' from ' + fname + ' slice=' + str(i) + ' has some problem'
        #     continue
        d = DICE(gg, ss)
        jac = JAC(gg, ss)
        gce = GCE(gg, ss)
        vs = VS(gg, ss)
        imindex = np.argmax(seg._itemindexes > i)
        data = pd.DataFrame([[os.path.basename(seg.dataSourcePath[imindex - 1]), imindex, i, gce, jac, d, 1-vs, int(cat.data.numpy())]],
                            columns=['filename','ImageIndex','SLICE', 'GCE', 'JAC', 'DICE', 'VD','Catagory'])
        df = df.append(data)
    return df


if __name__ == '__main__':
    seg = ImageDataSet("../ERA_Segmentation/03_TEST/true_postprocessed/", dtype=np.uint8, verbose=True)
    seg_nocat = ImageDataSet("../ERA_Segmentation/03_TEST/postprocessed_nocat/", dtype=np.uint8, verbose=True)
    gt = ImageDataSet("../ERA_Segmentation/03_TEST/gt", dtype=np.uint8, verbose=True)

    by_casedd = pd.DataFrame(columns=['filename', 'UseCategory','GCE', 'JAC', 'DICE', 'VD'])
    # np.set_printoptions(5, linewidth=200)
    # bycase = {'nocat':[], 'cat':[]}
    for i, row in enumerate(zip(seg, seg_nocat, gt)):
        funcs = [GCE, JAC, DICE, lambda seg, tar: 1-VS(seg, tar)]
        l_df = pd.DataFrame([[os.path.basename(seg.dataSourcePath[i]), 1] + [f(row[2].numpy(), row[0].numpy()) for f in funcs],
                             [os.path.basename(seg.dataSourcePath[i]), 0] + [f(row[2].numpy(), row[1].numpy()) for f in funcs],
                             ], columns=['filename', 'UseCategory', 'GCE', 'JAC', 'DICE', 'VD'], index=[i, i])
        by_casedd = by_casedd.append(l_df)
    by_casedd = by_casedd.sort_values('UseCategory')
    #
    seg = ImageDataSet("../ERA_Segmentation/03_TEST/true_postprocessed/", dtype=np.uint8, verbose=True, loadBySlices=0)
    seg_nocat = ImageDataSet("../ERA_Segmentation/03_TEST/postprocessed_nocat/", dtype=np.uint8, verbose=True, loadBySlices=0)
    gt = ImageDataSet("../ERA_Segmentation/03_TEST/gt", dtype=np.uint8, verbose=True, loadBySlices=0)
    # # seg.LoadWithCatagories("../ERA_Segmentation/03_TEST/CaseSegment_Test.txt")
    # # seg_nocat.LoadWithCatagories("../ERA_Segmentation/03_TEST/CaseSegment_Test.txt")
    gt.LoadWithCatagories("../ERA_Segmentation/03_TEST/CaseSegment_Test.txt")
    seg.LoadWithCatagories("../ERA_Segmentation/03_TEST/output.csv")
    seg_nocat.LoadWithCatagories("../ERA_Segmentation/03_TEST/output.csv")
    # gt.LoadWithCatagories("../ERA_Segmentation/03_TEST/output.csv")


    ramris = pd.read_csv("../ERA_Segmentation/03_TEST/list.csv")

    dd = EVAL(seg, gt)
    dd_nocat = EVAL(seg_nocat, gt)
    dd['UseCategory'] = 1
    dd_nocat['UseCategory'] = 0
    dd_nocat['Catagory'] = 0
    dd = pd.merge(ramris,dd)
    dd_nocat = pd.merge(ramris,dd_nocat)
    by_casedd = pd.merge(ramris, by_casedd)
    final = dd.append(dd_nocat)
    final.to_csv("../ERA_Segmentation/03_TEST/true_results.csv")
    by_casedd.to_csv("../ERA_Segmentation/03_TEST/true_results_bycase.csv")
    print final.to_string()
    print final['DICE'][final['Catagory']==2].mean()


