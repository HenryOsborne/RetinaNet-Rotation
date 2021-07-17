import argparse
import collections
import os
import os.path as osp

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from load_data import build_tfrecord_loader
from configs import cfgs


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--tfrecord', help='tfrecord path', type=str, default='data/tiny/DOTA_train.tfrecord')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=20)
    parser.add_argument('--version', type=str, default='TINY')

    args = parser.parse_args()

    return args


class TrainDOTA(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.args = parse_args()
        self.device = torch.device(cfgs.DEVICE)
        self.work_dir = osp.join('work_dir', self.args.version)
        os.makedirs(self.work_dir, exist_ok=True)

        self.loader = build_tfrecord_loader(self.args.tfrecord, self.cfgs.BATCH_SIZE)
        self.model = model.build_model(self.args.depth, cfgs, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfgs.LR, weight_decay=cfgs.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.model.train()
        self.model.freeze_bn()

    def train(self):
        loss_hist = collections.deque(maxlen=500)
        # retinanet.train()
        # retinanet.module.freeze_bn()

        for epoch_num in range(self.args.epochs):
            epoch_loss = []
            for iter_num, data in enumerate(self.loader):
                self.optimizer.zero_grad()
                img_name, img, gtboxes_and_label_r, gtboxes_and_label_h = data
                img = img.to(self.device)
                gtboxes_and_label_r = gtboxes_and_label_r[0].to(self.device)
                gtboxes_and_label_h = gtboxes_and_label_h[0].to(self.device)

                loss_dict = self.model((img, gtboxes_and_label_r, gtboxes_and_label_h))

                classification_loss = loss_dict['cls_loss']
                regression_loss = loss_dict['reg_loss']

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                self.optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running '
                    'loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss),
                                           np.mean(loss_hist)))

                del classification_loss
                del regression_loss

            self.scheduler.step(np.mean(epoch_loss))
            checkpoint = {
                'epoch': epoch_num,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(checkpoint,
                       osp.join(self.work_dir, '{}_retinanet_{}.pth'.format(self.cfgs.DATASET_NAME, epoch_num + 1)))


if __name__ == '__main__':
    trainer = TrainDOTA(cfgs)
    trainer.train()
