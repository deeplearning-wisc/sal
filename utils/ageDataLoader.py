from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import numpy as np
import os
from PIL import Image

'''
 data format:
 0001.jpg, 0002.jpg, 1,  0
 0034.jpg, 0023.jpg, -1, 1
 0022.jpg, 0002.jpg, 0,  1
 0001.jpg, 0045.jpg, 1,  2
 if binary == True, label1==-1 will be set 0.
 '''

class PairwiseDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, binary=False,
                 load_img=True, load_rgb=False, attr_num=10, root_dir='./data'):
        self.transform = transform
        self.target_transform = target_transform
        self.binary = binary
        self.root_dir = root_dir
        self.load_img = load_img
        self.load_rgb = load_rgb
        self.attr_num = attr_num
        self.lens = 0
        self.lines = list()

    @abstractmethod
    def _parse_each_item(self, line):
        pass

    @abstractmethod
    def _get_img_name(self, img_id):
        pass

    def __getitem__(self, index):
        img_id0, img_id1, label, attr_id, pair_id, strength = self._parse_each_item(
            self.lines[index])
        label, attr_id, pair_id, strength = np.float32(label), np.float32(
            attr_id), np.float32(pair_id), np.float32(strength)
        if self.binary:  # default is False
            if int(label) == -1:
                label = np.float32(0.)

        if self.load_img:
            img0 = Image.open(self._get_img_name(img_id0))
            img1 = Image.open(self._get_img_name(img_id1))
            if self.load_rgb:
                img0 = img0.convert('RGB')
                img1 = img1.convert('RGB')

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            img_id0, img_id1 = np.float32(img_id0), np.float32(img_id1)
            return img_id0, img_id1, img0, img1, label, attr_id, pair_id, strength
        else:
            img_id0, img_id1 = np.float32(img_id0), np.float32(img_id1)
            return img_id0, img_id1, label, attr_id, pair_id, strength

    def __len__(self):
        return self.lens


class AgeDataSet(PairwiseDataset):
    def __init__(self, txt, file2id_dict, id2file_dict, transform=None,
                 target_transform=None, binary=False, load_img=True, load_rgb=True, attr_num=1, root_dir='./data/'):
        super(AgeDataSet, self).__init__(txt, transform,
                                         target_transform, binary, load_img, load_rgb, attr_num, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.file2id_dict = file2id_dict
        self.id2file_dict = id2file_dict

    def _parse_each_item(self, line):
        split = line.split(',')
        img_id0, img_id1 = self.file2id_dict[split[0]
                                             ], self.file2id_dict[split[1]]
        label = int(split[2])  # comp_value
        pair_id = int(split[3])  # edge(x, y)_id
        strength = int(split[4])  # strength
        attr_id = 0  # attr_id
        return img_id0, img_id1, label, attr_id, pair_id, strength

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.id2file_dict[img_id])

    def get_img_file_name(self, img_id):
        return self.id2file_dict[img_id]


class PairwiseImgDataSet(Dataset):

    def __init__(self, txt, load_rgb=False, transform=None, target_transform=None, root_dir='./data'):
        self.load_rgb = load_rgb
        self.transform = transform
        self.target_transform = target_transform
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
            img_id1, img_id2, attr_id = self._parse_each_item(self.lines[i])
            img_ids.add((img_id1, attr_id))
            img_ids.add((img_id2, attr_id))
        self.img_ids = list(img_ids)

    def __getitem__(self, index):
        img_id, attr_id = self.img_ids[index]
        img = Image.open(self._get_img_name(img_id))
        if self.load_rgb:
            img = img.convert('RGB')
        img_id, attr_id = np.float32(img_id), np.float32(attr_id)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_id, attr_id

    def __len__(self):
        return self.lens


class AgeImgDataSet(PairwiseImgDataSet):

    def __init__(self, txt, file2id_dict, id2file_dict, load_rgb=True,
                 transform=None, target_transform=None, root_dir='./data'):
        super(AgeImgDataSet, self).__init__(
            txt, load_rgb, transform, target_transform, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.file2id_dict = file2id_dict
        self.id2file_dict = id2file_dict
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        attr_id = 0
        img_id1, img_id2 = self.file2id_dict[split[0]], self.file2id_dict[split[1]]
        return img_id1, img_id2, attr_id

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, self.id2file_dict[img_id])


def load_age(**kwargs):
    id2file_dict = dict()
    file2id_dict = dict()
    for file in os.listdir(kwargs['img_path']):
        img_id = file.split('.')[0].replace('A', '').replace(
            'a', '1').replace('b', '2').replace('c', '3')
        id2file_dict[int(img_id)] = file
        file2id_dict[file] = int(img_id)
    
    train_dataset = AgeDataSet(txt=kwargs['train_path'],
                            root_dir=kwargs['img_path'],
                            file2id_dict=file2id_dict,
                            id2file_dict=id2file_dict,
                            binary=False,
                            load_img=True, 
                            load_rgb=True,
                            transform=kwargs['transform'])
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=kwargs['batch_size'],
                                shuffle=kwargs['shuffle'], **kwargs['others'])

    img_dataset = AgeImgDataSet(txt=kwargs['test_path'], 
                                root_dir=kwargs['img_path'],
                                file2id_dict=file2id_dict,
                                id2file_dict=id2file_dict,
                                load_rgb=True,
                                transform=kwargs['transform'])
    img_loader = DataLoader(dataset=img_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False, **kwargs['others'])

    
    test_dataset_wo_img = AgeDataSet(txt=kwargs['test_path'],
                            root_dir=kwargs['img_path'],
                            file2id_dict=file2id_dict,
                            id2file_dict=id2file_dict,
                            binary=False,
                            load_img=False)
    test_loader_wo_img = DataLoader(dataset=test_dataset_wo_img,
                                batch_size=kwargs['batch_size'],
                                shuffle=False, **kwargs['others'])
    
    return train_dataset, train_loader, img_dataset, img_loader, test_dataset_wo_img, test_loader_wo_img


def get_age(img_id):
    splitor = None
    for i in img_id:
        if i < '0' or i > '9':
            splitor = i
            break
    _, left = img_id.split(splitor, 1)
    for i in left:
        if i < '0' or i > '9':
            splitor = i
            break
    age, left = left.split(splitor, 1)
    return int(age)