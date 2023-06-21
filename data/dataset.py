import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.common import read_annotations
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, annotations, config, balance=False, test_mode=False):
        self.resize_size = config.resize_size
        self.class_num = config.class_num
        self.balance = balance
        self.test_mode=test_mode
        self.config = config

        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if balance:
            self.data = [[x for x in annotations if x[1] == lab] for lab in [i for i in range(config.class_num)]]
        else:
            self.data = [annotations]
        
    def __len__(self):
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        if self.balance:
            labs = []
            imgs = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path, lab = self.data[i][safe_idx]
                img = self.load_sample(img_path)

                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)

            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                   torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            img = self.load_sample(img_path)
            lab = torch.tensor(lab, dtype=torch.long)

            return img, lab, img_path

    def load_sample(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if img.size[0]!=img.size[1]:
            short_size = img.size[0] if img.size[0]<img.size[1] else img.size[1]
            img = transforms.CenterCrop(size=(short_size, short_size))(img)

        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        img = self.norm_transform(img)

        return img


class BaseData(object):
    def __init__(self, train_data_path, val_data_path, 
                test_data_path, out_data_path, 
                opt, config):

        train_set = ImageDataset(read_annotations(train_data_path,opt.debug), config, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        
        val_set = ImageDataset(read_annotations(val_data_path,opt.debug), config, balance=False, test_mode=True)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

        tsne_set = ImageDataset(read_annotations(test_data_path,opt.debug), config, balance=True, test_mode=True)
        tsne_loader = DataLoader(
            dataset=tsne_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        test_set = ImageDataset(read_annotations(test_data_path,opt.debug), config, balance=False, test_mode=True)
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

        out_set = ImageDataset(read_annotations(out_data_path,opt.debug), config, balance=False, test_mode=True)
        out_loader = DataLoader(
            dataset=out_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )


        out_set1 = ImageDataset(read_annotations(out_data_path.replace('out','out_seed'),opt.debug), config, balance=False, test_mode=True)
        out_loader1 = DataLoader(
            dataset=out_set1,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

        out_set2 = ImageDataset(read_annotations(out_data_path.replace('out','out_arch'),opt.debug), config, balance=False, test_mode=True)
        out_loader2 = DataLoader(
            dataset=out_set2,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

        out_set3 = ImageDataset(read_annotations(out_data_path.replace('out','out_dataset'),opt.debug), config, balance=False, test_mode=True)
        out_loader3 = DataLoader(
            dataset=out_set3,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )
        self.out_loader1 = out_loader1
        self.out_loader2 = out_loader2
        self.out_loader3 = out_loader3

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.out_loader = out_loader
        self.tsne_loader = tsne_loader

        print('train: {}, val: {}, test {}, out {}'.format(len(train_set),len(val_set),len(test_set),len(out_set)))