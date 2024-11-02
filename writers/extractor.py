import zipfile
import os
from tqdm import tqdm


def extract_data_from_zip(zip_path: str, output_path: str):
    with zipfile.ZipFile(zip_path) as zipped:
        file_path_list = list(
            filter(lambda x: '.csv' in x or '.jpg' in x or '.jpeg' in x or '.png' in x, zipped.namelist()))

        for file in file_path_list:
            zipped.extract(file, output_path)


def extract_data_from_folder(folder_path, output_path):
    files = os.listdir(folder_path)

    for file in tqdm(files):
        print(file, 'file')
        extract_data_from_zip(f'{folder_path}/{file}', output_path)
