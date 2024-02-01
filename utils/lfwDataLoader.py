import sys 
sys.path.append("..") 

from torch.utils.data import Dataset, DataLoader
from utils.ageDataLoader import PairwiseDataset, PairwiseImgDataSet
import os

class LFWDataSet(PairwiseDataset):

    def __init__(self, txt, transform=None, target_transform=None, binary=False,
                 load_img=True, load_rgb=False, attr_num=10, root_dir='./data/images/'):
        super(LFWDataSet, self).__init__(txt, transform, target_transform,
                                         binary, load_img, load_rgb, attr_num, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)

    def _parse_each_item(self, line):
        split = line.split(',')
        img_id0, img_id1 = int(split[0].split(
            '.')[0]), int(split[1].split('.')[0])
        label = int(split[2])  # comp_value
        attr_id = int(split[3])  # attr_id
        pair_id = int(split[4])  # edge(x, y)_id
        strength = int(split[5])  # strength
        return img_id0, img_id1, label, attr_id, pair_id, strength

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, str(img_id) + '.jpg')

class LFWImgDataSet(PairwiseImgDataSet):

    def __init__(self, txt, load_rgb=False, transform=None, target_transform=None, root_dir='./data/images/'):
        super(LFWImgDataSet, self).__init__(
            txt, load_rgb, transform, target_transform, root_dir)
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self._get_all_img_id()
        self.lens = len(self.img_ids)

    def _parse_each_item(self, line):
        split = line.split(',')
        attr_id, img_id1, img_id2 = int(split[3]), int(
            split[0].split('.')[0]), int(split[1].split('.')[0])
        return img_id1, img_id2, attr_id

    def _get_img_name(self, img_id):
        return os.path.join(self.root_dir, str(img_id) + '.jpg')

def load_lfw(**kwargs):
    train_dataset = LFWDataSet(txt=kwargs['train_path'],
                            root_dir=kwargs['img_path'],
                            binary=False,
                            load_img=True,
                            load_rgb=True,
                            transform=kwargs['transform'])
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=kwargs['shuffle'], **kwargs['others'])

    img_dataset = LFWImgDataSet(txt=kwargs['test_path'], 
                                root_dir=kwargs['img_path'],
                                load_rgb=True,
                                transform=kwargs['transform'])
    img_loader = DataLoader(dataset=img_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False, **kwargs['others'])

    # test loader
    test_dataset_wo_img = LFWDataSet(txt=kwargs['test_path'],
                                        root_dir=kwargs['img_path'],
                                        binary=False,
                                        load_img=False,
                                        transform=kwargs['transform'])
    test_loader_wo_img = DataLoader(dataset=test_dataset_wo_img,
                            batch_size=kwargs['batch_size'],
                            shuffle=False, **kwargs['others'])

    return train_dataset, train_loader, img_dataset, img_loader, test_dataset_wo_img, test_loader_wo_img
