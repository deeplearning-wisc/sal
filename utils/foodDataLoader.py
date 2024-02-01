from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import numpy as np
import os
from PIL import Image
import pickle

class TripletDataset(Dataset):
    def __init__(self, transform=None, load_img=True, load_rgb=False, root_dir='../data'):
        self.transform = transform
        self.load_img = load_img
        self.load_rgb = load_rgb
        self.root_dir = root_dir
        self.lens = 0
        self.lines = list()

    @abstractmethod
    def _parse_each_item(self, line):
        pass

    @abstractmethod
    def _get_img_name(self, img_id):
        pass

    def __getitem__(self, index):
        anchor_id, pos_id, neg_id, weight = self._parse_each_item(self.lines[index])
        anchor_id, pos_id, neg_id, weight = np.int32(anchor_id), np.int32(pos_id), np.int32(neg_id), np.int32(weight)

        if self.load_img:
            anchor = Image.open(self._get_img_name(anchor_id))
            pos = Image.open(self._get_img_name(pos_id))
            neg = Image.open(self._get_img_name(neg_id))
            
            if self.load_rgb:
                anchor = anchor.convert('RGB')
                pos = pos.convert('RGB')
                neg = neg.convert('RGB')
                    
            if self.transform is not None:
                anchor = self.transform(anchor)
                pos = self.transform(pos)
                neg = self.transform(neg)
            
            return anchor_id, pos_id, neg_id, anchor, pos, neg ,weight
        else:
            return anchor_id, pos_id, neg_id, weight

    def __len__(self):
        return self.lens


class TripletImgDataSet(Dataset):
    
    def __init__(self, txt, load_rgb=False, transform=None, root_dir='../data'):
        self.load_rgb = load_rgb
        self.transform = transform
        self.root_dir = root_dir
        self.lines = list()
        self.lens = 0
    
    @abstractmethod
    def _parse_each_item(self, line):
        pass
    
    @abstractmethod
    def _get_img_name(self, img_id):
        pass
    
    def _get_all_img_id(self):
        img_ids = set()
        for i in range(len(self.lines)):
            anchor_id, pos_id, neg_id = self._parse_each_item(self.lines[i])
            img_ids.add(anchor_id)
            img_ids.add(pos_id)
            img_ids.add(neg_id)
        self.img_ids = list(img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = Image.open(self._get_img_name(img_id))
        
        if self.load_rgb:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        img_id = np.int32(img_id)

        return img, img_id, 0
    
    def __len__(self):
        return self.lens

class FoodDataSet(TripletDataset):
    def __init__(self, txt, file_dict, transform=None, load_img=True, load_rgb=True, root_dir='../data'):
        super(FoodDataSet, self).__init__(transform, load_img, load_rgb, root_dir)
        
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.file_dict = file_dict

    def _parse_each_item(self, line):
        split = line.split(',')
        anchor_id, pos_id, neg_id = int(split[0]), int(split[1]), int(split[2])
        weight = int(split[3])  # strength
        return anchor_id, pos_id, neg_id, weight

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.file_dict[img_id])


class FoodImgDataSet(TripletImgDataSet):
    
    def __init__(self, txt, file_dict, load_rgb=False, transform=None, root_dir='../data'):
        super(FoodImgDataSet, self).__init__(txt, load_rgb, transform, root_dir)
        
        self.lines = [line.rstrip() for line in open(txt, 'r')]
            
        self.file_dict = file_dict
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        anchor_id, pos_id, neg_id = int(split[0]), int(split[1]), int(split[2])
        return anchor_id, pos_id, neg_id
        
    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.file_dict[img_id])

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def load_food(**kwargs):
    int2food = load_obj(kwargs['int2food_path'])

    train_dataset = FoodDataSet(txt=kwargs['train_path'],
                            root_dir=kwargs['img_path'],
                            file_dict=int2food,
                            load_img=True, 
                            load_rgb=True,
                            transform=kwargs['transform'])
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=kwargs['batch_size'],
                                shuffle=kwargs['shuffle'], **kwargs['others'])

    img_dataset = FoodImgDataSet(txt=kwargs['test_path'], 
                                root_dir=kwargs['img_path'],
                                file_dict=int2food,
                                load_rgb=True,
                                transform=kwargs['transform'])
    img_loader = DataLoader(dataset=img_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False, **kwargs['others'])

    
    test_dataset_wo_img = FoodDataSet(txt=kwargs['test_path'],
                            root_dir=kwargs['img_path'],
                            file_dict=int2food,
                            load_img=False)
    test_loader_wo_img = DataLoader(dataset=test_dataset_wo_img,
                                batch_size=kwargs['batch_size'],
                                shuffle=False, **kwargs['others'])
    
    return train_dataset, train_loader, img_dataset, img_loader, test_dataset_wo_img, test_loader_wo_img

