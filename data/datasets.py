from data import singlecoil_data
import numpy as np
from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, phase, dataset, domain_1, domain_2, transform=None):
        super().__init__()
        self.transform = transform
        self.phase = phase
        self.fs_images_domain_1, self.fs_images_domain_2 = singlecoil_data.get_synthesis_dataset_both(phase, dataset, domain_1, domain_2, norm=False)

    def __len__(self):
        return len(self.fs_images_domain_1)

    def __getitem__(self, idx):
        if self.transform is not None:
           return self.transform(torch.from_numpy(self.fs_images_domain_1[np.newaxis,idx])),self.transform(torch.from_numpy(self.fs_images_domain_2[np.newaxis,idx]))
        else:
            dom1_im = self.fs_images_domain_1[idx]
            dom2_im = self.fs_images_domain_2[idx]
        return dom1_im[np.newaxis], dom2_im[np.newaxis]
    
    
class ImageDataset_train_single_domain(Dataset):
    def __init__(self, dataset, phase, contrast, transform=None):
        super().__init__()
        self.transform = transform
        self.phase = phase
        self.contrast = contrast
        self.dataset =  dataset
        self.images, self.labels, self.ys  = singlecoil_data.get_single_domain_dataset(phase, contrast, dataset=dataset, norm=False)

        print("total number of images in training: ", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(torch.from_numpy(self.images[idx])), self.labels[idx][0], self.ys[idx] 
        else: 
            return self.images[idx], self.labels[idx][0], self.ys[idx]        

class ImageDataset_train_multi_domain(Dataset):
    def __init__(self, dataset, phase, transform=None):
        super().__init__()
        self.transform = transform
        self.phase = phase
        self.dataset =  dataset
        self.images, self.labels, self.ys = singlecoil_data.get_multi_domain_dataset(phase, dataset=dataset, norm=False)

        print("total number of images in training: ", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(torch.from_numpy(self.images[idx])), self.labels[idx][0], self.ys[idx]
        else: 
            return self.images[idx], self.labels[idx][0], self.ys[idx]



def prepare_datasets_cyclegan_multi(dataset, batch_size, augmentation):
    domain_1_dataset = ImageDataset_train_multi_domain(dataset=dataset,phase="train", transform=augmentation)
    domain_2_dataset = ImageDataset_train_multi_domain(dataset=dataset,phase="train", transform=augmentation)

    sampler_train_1 = torch.utils.data.RandomSampler(domain_1_dataset)
    sampler_train_2 = torch.utils.data.RandomSampler(domain_2_dataset)
    data_loader_train_1 = torch.utils.data.DataLoader(
    domain_1_dataset, sampler=sampler_train_1,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    )
    data_loader_train_2 = torch.utils.data.DataLoader(
        domain_2_dataset, sampler=sampler_train_2,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    return data_loader_train_1, data_loader_train_2



def prepare_datasets_cyclegan_single(domain_1, domain_2, dataset, batch_size, augmentation):
    domain_1_dataset = ImageDataset_train_single_domain(dataset,"train",domain_1, transform=augmentation)
    domain_2_dataset = ImageDataset_train_single_domain(dataset,"train",domain_2, transform=augmentation)
    sampler_train_1 = torch.utils.data.RandomSampler(domain_1_dataset)
    sampler_train_2 = torch.utils.data.RandomSampler(domain_2_dataset)
    
    data_loader_train_1 = torch.utils.data.DataLoader(
    domain_1_dataset, sampler=sampler_train_1,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    )
    data_loader_train_2 = torch.utils.data.DataLoader(
        domain_2_dataset, sampler=sampler_train_2,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return data_loader_train_1, data_loader_train_2