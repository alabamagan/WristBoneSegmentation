from torch.utils.data import Dataset

class ImageFeaturePair(Dataset):
    """
    Data set wrapping like Tensor Dataset, except this also accept a mask.
    """

    def __init__(self, image_Dataset, landmarkDataset):
        assert len(image_Dataset) == len(landmarkDataset)

        self.image_dataset = image_Dataset
        self.landmarks_dataset = image_Dataset

    def __getitem__(self, index):
        return self.image_dataset[index], self.landmarks_dataset[index]

    def __len__(self):
        return len(self.image_dataset)

    def __str__(self):
        print str(self.image_dataset) + str(self.landmarks_dataset)