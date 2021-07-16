import torch
from retinanet import model
from load_data import build_tfrecord_loader

tfrecord_path = "./data/tiny/DOTA_train.tfrecord"
loader = build_tfrecord_loader(tfrecord_path)
retina = model.resnet50(50, pretrained=True)
retina.eval()

for i, (img_name, img, gtboxes_and_label_r, gtboxes_and_label_h) in enumerate(loader):
    out = retina(img, gtboxes_and_label_r, gtboxes_and_label_h)
print(1)
