import argparse
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

from classifier.get_class_model import get_class_model
from ssformer.get_ssformer_model import get_ssformer_model


def make_predict_csv(pic_path, val_csv_path):
    data_list = []
    class_df = pd.DataFrame(columns=["id", "path", "class_predict"])
    if os.path.exists(os.path.join(pic_path, 'test')):
        path_root = os.path.join(pic_path, 'test')
        for item_case in os.listdir(path_root):
            for item_day in os.listdir(os.path.join(path_root, item_case)):
                path = os.path.join(path_root, item_case, item_day, 'scans')
                data_list.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
        class_df["path"] = data_list
        class_df["id"] = class_df["path"].apply(lambda x: str(x.split("/")[5]) + "_" + str(
            x.split("/")[-1].split("_")[0] + '_' + x.split("/")[-1].split("_")[1]))
        pre = True
    else:
        val_csv = pd.read_csv(val_csv_path)
        class_df[["id", "path", "class"]] = val_csv[["id", "path", "classes"]]
        pre = False
    return class_df, pre


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


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


def Pre_pic(pic_path, pre, data_transform):
    if pre:
        png = cv2.imread(pic_path)
        if not (png == 0).all():
            png = png * 5
            png[png > 255] = 255
            png = gamma_trans(png, math.log10(0.5) / math.log10(np.mean(png[png > 0]) / 255))
        image = Image.fromarray(cv2.cvtColor(png, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(pic_path)
    size = (image.size[1], image.size[0])
    return data_transform(image), size


def main(args):
    # get devices
    if torch.cuda.is_available():
        torch.cuda.set_device(args.GPU)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("using {} device.".format(device))

    # class_model
    model_class = get_class_model(num_classes=2, backbone=args.model_class_name, pretrained=False)
    model_class.load_state_dict(torch.load(args.class_weights_path, map_location='cpu')['model'])
    model_class.to(device)
    model_class.eval()
    # SS_former
    model_seg = get_ssformer_model(model_name=args.model_seg_name, num_classes=args.num_classes + 1)
    model_seg.load_state_dict(torch.load(args.seg_weights_path, map_location='cpu')['model'])
    model_seg.to(device)
    model_seg.eval()

    # 获取预测csv
    class_df, pre = make_predict_csv(args.pic_path, args.val_csv_path)

    # 生成提交csv
    sub_df = pd.DataFrame(columns=["id", "class", "predicted"])
    class_dict = dict(zip([0, 1, 2], ['large_bowel', 'small_bowel', 'stomach']))

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean[0], std=std[0]),
                                         transforms.Resize((args.size, args.size))
                                         ])

    # 开始预测
    print(args)
    with tqdm(total=len(class_df), mininterval=0.3) as pbar:
        for item in range(len(class_df)):
            with torch.no_grad():
                image, size = Pre_pic(class_df.loc[item, 'path'], pre, data_transform)
                # expand batch dimension
                image = torch.unsqueeze(image, dim=0).to(device)
                if model_class(image)[0].argmax().item() == 1:
                    class_df.loc[item, "class_predict"] = 1
                    prediction = model_seg(image)['out']
                    predictions = F.resize(torch.stack(
                        [prediction[0][[0, item + 1], ...].argmax(0) for item in range(args.num_classes)], dim=0),
                        size, interpolation=transforms.InterpolationMode.NEAREST).permute(1, 2, 0).cpu().numpy()
                    for item_class in range(predictions.shape[-1]):
                        if not (predictions[..., item_class] == 0).all():
                            list_item = predictions[..., item_class]
                            list_item[list_item != 0] = 1
                            sub_df.loc[item] = [class_df.loc[item, 'id'], class_dict[item_class], rle_encode(list_item)]
                        else:
                            sub_df.loc[item] = [class_df.loc[item, 'id'], class_dict[item_class], ""]
                else:
                    class_df.loc[item, "class_predict"] = 0
                    sub_df.loc[item] = [class_df.loc[item, 'id'], class_dict[item_class], ""]
            pbar.update()

    # 生成submission.csv
    if os.path.exists(os.path.join(args.pic_path, 'test')):
        df_ssub = pd.read_csv(os.path.join(args.pic_path, 'sample_submission.csv'))
        del df_ssub['predicted']
        sub_df = df_ssub.merge(sub_df, on=['id', 'class'])
        assert len(sub_df) == len(df_ssub)
    else:
        print(len(class_df.loc[((class_df['class'] == 0) & (class_df['class_predict'] == 0)).index.tolist(), :]) /
              len(class_df.loc[class_df['class'] == 0, :]))
        print(len(class_df.loc[((class_df['class'] != 0) & (class_df['class_predict'] != 0)).index.tolist(), :]) /
              len(class_df.loc[class_df['class'] != 0, :]))
        class_df.to_csv(os.path.join(args.save_dir, 'class_predict.csv'), index=False)
    sub_df[['id', 'class', 'predicted']].to_csv(os.path.join(args.save_dir, 'submission.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='submit set')
    parser.add_argument('--GPU', type=int, default=0, help='GPU_ID')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--size', type=int, default=384, help='pic size')
    parser.add_argument('--model_class_name', type=str, default='resnet18')
    parser.add_argument('--model_seg_name', type=str, default='mit_PLD_b2')
    parser.add_argument('--class_weights_path', type=str)
    parser.add_argument('--seg_weights_path', type=str)
    parser.add_argument('--pic_path', type=str, default=r"../input/uw-madison-gi-tract-image-segmentation",
                        help="pic文件夹位置")
    parser.add_argument('--val_csv_path', type=str,
                        default=r"../input/uw-all-classes-csv/val_csv.csv", help='预测csv路径')
    parser.add_argument('--save_dir', type=str, default="./", help='存储文件夹位置')
    args = parser.parse_args()

    main(args)




