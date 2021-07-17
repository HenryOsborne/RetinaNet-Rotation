import argparse
import collections
import os
import os.path as osp

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from load_data import DotaTest
from configs import cfgs


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--show_box', '-s', default=False, action='store_true')
    parser.add_argument('--val_image', help='Path to val image', type=str, default='data/tiny/images')
    parser.add_argument('--model', help='Path to model (.pth) file.', type=str,
                        default='work_dir/TINY/DOTA_retinanet_20.pth')

    args = parser.parse_args()

    return args


class TestDOTA(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        self.device = torch.device(cfgs.DEVICE)
        self.work_dir = osp.join('work_dir', cfgs.VERSION)
        os.makedirs(self.work_dir, exist_ok=True)

        self.model = model.build_model(self.args.depth, cfgs, self.device).to(self.device)
        checkpoint = torch.load(self.args.model)
        self.model.load_state_dict(checkpoint['model'])
        print('successful loading {}...'.format(self.args.model.split('/')[-1]))

        self.dataset = DotaTest(self.args.val_image, self.cfgs)
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfgs.BATCH_SIZE_VAL,
                                     collate_fn=self.dataset.collate_fn_val)

        self.model.eval()

    @torch.no_grad()
    def eval(self):
        for i, (img, image_name) in enumerate(self.dataloader):
            print(image_name)
            img = img.to(self.device)

            boxes, scores, category = self.model(img)


if __name__ == '__main__':
    tester = TestDOTA(cfgs)
    tester.eval()
