import torch
import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


#print(ds.train_labels[indices_3])
#print(ds.train_data[indices_3])

class CroppedRescaledMNISTDataset(Dataset):
    def __init__(self, labels, thresholds): 
        super(Dataset, self).__init__()
        self.ds = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((4, 4), interpolation=2),
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.first_label = labels[0]
        self.second_label = labels[1]
        self.indices_3 = self.ds.targets==self.first_label
        self.indices_6 = self.ds.targets==self.second_label
        self.train_data_3 = [self.ds[idx] for idx in range(60000) if self.indices_3[idx] == True] 
        self.train_data_6 = [self.ds[idx] for idx in range(60000) if self.indices_6[idx] == True] 

        #compute means of the two labels
        self.mean_3 = torch.zeros(self.train_data_3[0][0].shape)
        self.mean_6 = torch.zeros(self.train_data_6[0][0].shape)
        for sample in self.train_data_3:
            self.mean_3 = self.mean_3 + sample[0]

        for sample in self.train_data_6:
            self.mean_6 = self.mean_6 + sample[0]
        
        self.mean_3 = self.mean_3 / self.train_data_3.__len__()
        self.mean_6 = self.mean_6 / self.train_data_6.__len__()

        #optimize means
        self.mean_3 = self.mean_3 / self.mean_3.max()
        self.mean_3[self.mean_3 < 0.7] = 0
        self.mean_3[self.mean_3 >= 0.7] = 1
        self.mean_6 = self.mean_6 / self.mean_6.max()
        self.mean_6[self.mean_6 < 0.7] = 0
        self.mean_6[self.mean_6 >= 0.7] = 1
    
        #self.mean_3 = torch.zeros(self.train_data_3[0][0].shape)
        #self.mean_6 = torch.ones(self.train_data_6[0][0].shape)


        #Find clusters around means
        self.thresholds = thresholds
        self.diff_3 = [(sample[0] - self.mean_3).norm() for sample in self.train_data_3] 
        self.idx_3 = [idx for idx in range(self.diff_3.__len__()) if (self.diff_3[idx] < self.thresholds[0])]
        self.train_data_3 = [self.train_data_3[idx] for idx in self.idx_3]
        
        self.diff_6 = [(sample[0] - self.mean_6).norm() for sample in self.train_data_6] 
        self.idx_6 = [idx for idx in range(self.diff_6.__len__()) if (self.diff_6[idx] < self.thresholds[1])]
        self.train_data_6 = [self.train_data_6[idx] for idx in self.idx_6]

        plt.imshow(self.mean_3.squeeze(0))
        plt.show()

        plt.imshow(self.mean_6.squeeze(0))
        plt.show()
        plt.close()

        print("First Label Samples", self.train_data_3.__len__())
        print("Second Label Samples", self.train_data_6.__len__())
        self.data = self.train_data_3 + self.train_data_6
        
    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return [self.data[idx][0].view(16, 1).numpy(), self.data[idx][1]]


if __name__ == "__main__":
    batch_size = 10
    res_ds = CroppedRescaledMNISTDataset(labels=[3, 6], thresholds=[1, 0.92])
    print(len(res_ds))
    train_loader = torch.utils.data.DataLoader(res_ds, batch_size=batch_size, shuffle=True)

    it = iter(train_loader)
    x, y = next(it)
    print(x.shape)
    print(y)

    for sample in range(batch_size):
        sample = x[sample].squeeze(0)
        plt.imshow(sample)
        plt.show()
        plt.close()


