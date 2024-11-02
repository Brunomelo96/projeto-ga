import concurrent.futures
import os
import zipfile
from PIL import Image


def write_images(input_path, output_path):

    # sink = wds.TarWriter(output_path, encoder=False,)
    with zipfile.ZipFile(output_path, 'w') as zip:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for entry in os.scandir(input_path):
                _, pure_img_path = entry.path.rsplit('/', 1)

                executor.submit(write_image, pure_img_path,
                                input_path, zip)


def get_label_from_image(img_path: str):
    label, _ = img_path.rsplit('_', 1)
    _, label = label.rsplit('_', 1)

    return label


def get_image_name(img_path: str):
    name, _ = img_path.rsplit('.', 1)
    return name


def write_image(img_path, input_path, zip: zipfile.ZipFile):
    full_img_path = f'{input_path}/{img_path}'
    # label = get_label_from_image(img_path)
    img = Image.open(full_img_path)

    img_name = get_image_name(img_path)
    print(img_path)
    zip.writestr(f'{img_name}.jpeg', img.tobytes())
