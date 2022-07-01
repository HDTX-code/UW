import argparse
import copy
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

from classifier.get_class_model import get_class_model
from ssformer.build_ssformer_model import build

sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append('../input/openmmlab-essential-repositories/openmmlab-repos/mmcv')
sys.path.append('../input/addict/addict')


class TestdataSet(Dataset):
    def __init__(self, csv, pre, transform):
        self.csv = csv
        self.pre = pre
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, item):
        path = self.csv.loc[item, 'path']
        img, size = self.Pre_pic(path, self.pre, self.transform)
        return img, size, item

    def Pre_pic(self, pic_path, pre, data_transform):
        png = cv2.imread(pic_path)
        if pre:
            if not (png == 0).all():
                png = png * 5
                png[png > 255] = 255
                png = self.gamma_trans(png, math.log10(0.5) / math.log10(np.mean(png[png > 0]) / 255))
        image = Image.fromarray(cv2.cvtColor(png, cv2.COLOR_BGR2RGB))
        size = (image.size[1], image.size[0])
        return data_transform(image), size

    @staticmethod
    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    @staticmethod
    def collate_fn(batch):
        img, size, item = list(zip(*batch))
        image = torch.stack([i for i in img], dim=0)
        return image, size, item


def get_model(model_name, num_classes):
    model = build(model_name=model_name, class_num=num_classes)
    return model


def make_predict_csv(pic_path):
    data_list = []
    case_day_list = []
    class_df = pd.DataFrame(columns=["id", "path", "class_predict"])
    if os.path.exists(os.path.join(pic_path, 'test')):
        path_root = os.path.join(pic_path, 'test')
        case_list = os.listdir(path_root)
    else:
        path_root = os.path.join(pic_path, 'train')
        case_list = os.listdir(path_root)[:10]
    with tqdm(total=len(case_list)) as pbar:
        for item_case in case_list:
            for item_day in os.listdir(os.path.join(path_root, item_case)):
                path = os.path.join(path_root, item_case, item_day, 'scans')
                data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
                for len_item in os.listdir(path):
                    case_day_list.append(item_day)
            pbar.update(1)
    for i, item_pic_path in enumerate(data_list):
        class_df.loc[len(class_df)] = [case_day_list[i] + '_' + item_pic_path[-32:-22], item_pic_path, ""]
    class_df.index = list(range(len(class_df)))
    return class_df, True


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    # num_workers
    args.num_workers = min(min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]),
                           args.num_workers)

    # model初始化
    model = get_model(model_name=args.model_name, num_classes=args.num_classes + 1)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    # model初始化
    model_c = get_class_model(2, backbone=args.model_name_c, pretrained=False)
    model_c.load_state_dict(torch.load(args.weights_path_c, map_location='cpu')['model'])
    model_c.to(device)
    model_c.eval()

    # 获取预测csv
    class_df, pre = make_predict_csv(args.pic_path)

    # 生成提交csv
    sub_df = pd.DataFrame(columns=["id", "class", "predicted"])
    class_dict = ['stomach', 'small_bowel', 'large_bowel']

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std),
                                         transforms.Resize((args.size, args.size))
                                         ])

    # dataloader
    dataset_test = TestdataSet(copy.deepcopy(class_df), pre, data_transform)
    gen = torch.utils.data.DataLoader(dataset_test,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      collate_fn=dataset_test.collate_fn,
                                      shuffle=False)
    # 开始预测
    print(args)
    print(pre)
    with tqdm(total=len(gen), mininterval=0.3) as pbar:
        for item_img, item_size, item in gen:
            with torch.no_grad():
                cl = model_c(item_img.to(device)).argmax(1)
                prediction = model(item_img.to(device))['out']
                for item_batch in range(prediction.shape[0]):
                    if cl[item_batch].item() == 1:
                        predictions = F.resize(torch.stack(
                            [prediction[item_batch][[0, item_C + 1], ...].argmax(0) for item_C in range(args.num_classes)],
                            dim=0), item_size[item_batch],
                            interpolation=transforms.InterpolationMode.NEAREST).permute(1, 2, 0).cpu().numpy()
                        for item_class in range(predictions.shape[-1]):
                            if not (predictions[..., item_class] == 0).all():
                                list_item = predictions[..., item_class]
                                list_item[list_item != 0] = 1
                                sub_df.loc[len(sub_df)] = [class_df.loc[item[item_batch], 'id'], class_dict[item_class],
                                                           rle_encode(list_item)]
                            else:
                                sub_df.loc[len(sub_df)] = [class_df.loc[item[item_batch], 'id'], class_dict[item_class], ""]
                    else:
                        for item_class in range(3):
                            sub_df.loc[len(sub_df)] = [class_df.loc[item[item_batch], 'id'], class_dict[item_class], ""]
            pbar.update()

    # 生成submission.csv
    if os.path.exists(os.path.join(args.pic_path, 'test')):
        df_ssub = pd.read_csv(os.path.join(args.pic_path, 'sample_submission.csv'))
        del df_ssub['predicted']
        sub_df = df_ssub.merge(sub_df, on=['id', 'class'])
        assert len(sub_df) == len(df_ssub)
    else:
        df_ssub = pd.read_csv(os.path.join(args.pic_path, 'train.csv'))
        del df_ssub['segmentation']
        sub_df = df_ssub.merge(sub_df, on=['id', 'class'])

    sub_df[['id', 'class', 'predicted']].to_csv(os.path.join(args.save_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='submit set')
    parser.add_argument('--model_name', type=str, default='mit_PLD_b4')
    parser.add_argument('--model_name_c', type=str, default='resnet18')
    parser.add_argument('--size', type=int, default=384, help='pic size')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--num_workers', type=int, default=24, help="num_workers")
    parser.add_argument('--weights_path', default='weights/loss_20220629202632/best_model_mit_PLD_b4.pth', type=str,
                        help='training weights')
    parser.add_argument('--weights_path_c', default='weights/loss_20220629202632/best_model_mit_PLD_b4.pth', type=str,
                        help='training weights')
    parser.add_argument('--pic_path', type=str, default=r"C:\Users\12529\Desktop\test",
                        help="pic文件夹位置")
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    args = parser.parse_args()

    main(args)
