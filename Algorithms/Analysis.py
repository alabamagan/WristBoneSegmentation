import numpy as np
import pandas as pd

from category_parser import *

def classification_accuracy(target, input):
    tarstack = []
    resstack = []
    for keys in tar:
        tarstack.extend(cat_list2slice_stack(tar[keys]))
        resstack.extend(cat_list2slice_stack(res[keys]))

    tarstack = np.array(tarstack)
    resstack = np.array(resstack)

    s = len(list(set(tarstack)))

    A_ac = (tarstack != 0) & (resstack != 0)
    pp = np.zeros([s, s])
    for i in xrange(s):
        for j in xrange(s):
            guess_i_is_j = sum(tarstack[resstack == i] == j)
            pp[i, j] = guess_i_is_j

    print pp, np.sum(pp)
    print sum(A_ac) / float(len(tarstack))
    return np.sum(tarstack == resstack) / float(len(tarstack))


if __name__ == '__main__':
    tar = categroy_file_reader('/home/lwong/Source/Repos/WraistBoneSegmentation/ERA_Segmentation/03_TEST/CaseSegment_Test.txt')
    res = categroy_file_reader('/home/lwong/Source/Repos/WraistBoneSegmentation/ERA_Segmentation/03_TEST/output.csv')
    # res = categroy_file_reader('/home/lwong/Source/Repos/WraistBoneSegmentation/ERA_Segmentation/output/results_new.csv')
    # tar = categroy_file_reader('/home/lwong/Source/Repos/WraistBoneSegmentation/ERA_Segmentation/output/new_output.csv')

    print classification_accuracy(tar, res)

