import cv2
import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
from configs import cfgs
from utils.image_augmentation import ImageAugmentation
from libs.utils.coordinate_convert import get_horizen_minAreaRectangle


def resize_image(image, bboxes, input_size=608):
    h, w, _ = image.shape  # (h,w,c)

    scale = input_size / max(w, h)  # 得到input size/图像的宽和高较小的那一个scale
    w, h = int(scale * w), int(scale * h)  # 将原图像resize到这个大小,不改变原来的形状

    image = cv2.resize(image, (w, h))
    fill_value = 0  # 选择边缘补空的像素值
    new_image = np.ones((input_size, input_size, 3)) * fill_value  # 新的符合输入大小的图像
    dw, dh = (input_size - w) // 2, (input_size - h) // 2
    new_image[dh:dh + h, dw:dw + w, :] = image

    # 将bbox也映射到resize后的坐标
    bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] * scale + dw
    bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] * scale + dh

    return new_image, bboxes


def decode_example(data):
    img_name = data['img_name'].tolist()
    img_name = ''.join([chr(int(i)) for i in img_name])
    img_height = int(data['img_height'])
    img_width = int(data['img_width'])
    num_objects = int(data['num_objects'])
    img = data['img'].reshape(img_width, img_height, 3)
    gtboxes_and_label = data['gtboxes_and_label'].tostring()
    gtboxes_and_label = np.fromstring(gtboxes_and_label, dtype=np.int32)
    gtboxes_and_label = gtboxes_and_label.reshape(num_objects, -1)

    img, gtboxes_and_label = resize_image(img, gtboxes_and_label)

    gtboxes_and_label = torch.from_numpy(gtboxes_and_label)
    img = torch.from_numpy(img).permute(2, 0, 1)
    data['img_name'] = img_name
    data['img_height'] = img_height
    data['img_width'] = img_width
    data['img'] = img
    data['num_objects'] = num_objects
    data['gtboxes_and_label'] = gtboxes_and_label
    return data


def augment(data):
    img = data['img'].float()
    gtboxes_and_label = data['gtboxes_and_label']
    image_augment = ImageAugmentation(cfgs)

    # if is_training:
    if cfgs.RGB2GRAY:
        img = image_augment.random_rgb2gray(img_tensor=img, gtboxes_and_label=gtboxes_and_label)

    if cfgs.IMG_ROTATE:
        img, gtboxes_and_label = image_augment.random_rotate_img(img_tensor=img,
                                                                 gtboxes_and_label=gtboxes_and_label)

    # img, gtboxes_and_label, img_h, img_w = image_augment.short_side_resize(img_tensor=img,
    #                                                                        gtboxes_and_label=gtboxes_and_label,
    #                                                                        target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
    #                                                                        length_limitation=cfgs.IMG_MAX_LENGTH)

    if cfgs.HORIZONTAL_FLIP:
        img, gtboxes_and_label = image_augment.random_flip_left_right(img_tensor=img,
                                                                      gtboxes_and_label=gtboxes_and_label)
    if cfgs.VERTICAL_FLIP:
        img, gtboxes_and_label = image_augment.random_flip_up_down(img_tensor=img,
                                                                   gtboxes_and_label=gtboxes_and_label)

    # else:
    #     img, gtboxes_and_label, img_h, img_w = \
    #         image_augment.short_side_resize(img_tensor=img,
    #                                         gtboxes_and_label=gtboxes_and_label,
    #                                         target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
    #                                         length_limitation=cfgs.IMG_MAX_LENGTH)

    mean = torch.as_tensor(cfgs.PIXEL_MEAN_)
    std = torch.as_tensor(cfgs.PIXEL_STD)
    img = (img / 255 - mean[:, None, None]) / std[:, None, None]

    data['img'] = img
    data['gtboxes_and_label_q'] = gtboxes_and_label
    data['gtboxes_and_label_h'] = get_horizen_minAreaRectangle(gtboxes_and_label, True)
    return data


def collate_fn(batch):
    for i in range(len(batch)):
        batch[i] = decode_example(batch[i])
        batch[i] = augment(batch[i])
    img_name = [i['img_name'] for i in batch]
    img_height = [i['img_height'] for i in batch]
    img_width = [i['img_width'] for i in batch]
    img = [i['img'] for i in batch]
    gtboxes_and_label_q = [i['gtboxes_and_label_q'] for i in batch]
    gtboxes_and_label_h = [i['gtboxes_and_label_h'] for i in batch]
    num_objects = [i['num_objects'] for i in batch]

    assert len(img_name) != 0
    img = torch.stack(img)  # 在新增的维度上合并tensor
    return tuple((img_name, img, gtboxes_and_label_q, gtboxes_and_label_h))


def build_tfrecord_loader(tfrecord_path, batch_size):
    index_path = None
    description = {"img_name": "byte", "img_height": "int", "img_width": "int",
                   "img": "byte", "gtboxes_and_label": "byte", "num_objects": "int"}
    dataset = TFRecordDataset(tfrecord_path, index_path, description)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader


if __name__ == '__main__':
    tfrecord_path = "./data/tiny/DOTA_train.tfrecord"
    loader = build_tfrecord_loader(tfrecord_path, 1)

    for i, data in enumerate(loader):
        print(data)
        print(1)
