from torch.utils.data import Dataset
from torch import from_numpy, cat
import os
import numpy as np
import pandas as pd


class Landmarks(Dataset):
    def __init__(self, csv_dir, dim=2):
        super(Landmarks, self).__init__()
        assert os.path.isfile(csv_dir), "Cannot open csv file!"
        self.root_dir = csv_dir
        self.data = []
        self.d = None
        self.length = 0
        self.dataReference = []
        self._ParseRootDir()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item]


    def _ParseRootDir(self):
        self.d = pd.read_csv(self.root_dir)
        for i, row in self.d.iterrows():
            f = row['File']
            r, g, b, y = [eval(row[keys]) for keys in ['Proximal Phalanx',
                                                       'Sesamoid',
                                                       'Metacarpal',
                                                       'Distal Phalanx']]
            self.dataReference.append(f)
            self.data.append(from_numpy(np.array([r, g, b, y], np.float)).unsqueeze(0))

        self.data = cat(self.data)
        self.length = self.data.size()[0]

    def __str__(self):
        s = "==========================================================================================\n" \
            "CSV source: %s \n" \
            "Number of loaded landmarkds: %i\n" \
            "Details:\n" \
            "--------------\n"%(self.root_dir, self.length)

        s += self.d.to_string()
        return s


if __name__ == '__main__':
    lm =  Landmarks("./TOCI/04.Resized/Landmarks.csv")
    print lm
    print lm[0:5]