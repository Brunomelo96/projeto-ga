import os
import random
from PIL import Image
import webdataset as wds


def get_label_from_image(img_path: str):
    label, _ = img_path.rsplit('_', 1)
    _, label = label.rsplit('_', 1)

    return label


def get_image_name(img_path: str):
    name, _ = img_path.rsplit('.', 1)
    return name


def write_images(input_path, train_path, test_path, test_size=.2, max_count=None):
    count = 0
    for entry in os.scandir(input_path):
        _, pure_img_path = entry.path.rsplit('/', 1)
        pick = random.random()

        if max_count is not None and count > max_count:
            return

        if pick > test_size:
            write_image(pure_img_path, input_path, train_path)
        else:
            write_image(pure_img_path, input_path, test_path)

        count += 1


def write_image(img_path: str, input_path: str, output_path: wds.TarWriter):
    full_img_path = f'{input_path}/{img_path}'
    label = int(get_label_from_image(img_path)) - 1
    img = Image.open(full_img_path)
    img = img.resize((224, 224))

    img_name = get_image_name(img_path)

    if not os.path.isdir(f'{output_path}/{label}'):
        os.mkdir(f'{output_path}/{label}')

    print(img_name)

    img.save(f'{output_path}/{label}/{img_name}.jpg')
