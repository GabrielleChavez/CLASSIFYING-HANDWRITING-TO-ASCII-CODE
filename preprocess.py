import os
from PIL import Image
import pandas as pd
import numpy as np

def get_data(data_dir: str, csv_dir: str) -> tuple:
    data_csv = pd.read_csv(csv_dir)

    X = []
    y = []

    for _, row in data_csv.iterrows():
        mapping = row.iloc[0]

        folder_dir = os.path.join(data_dir, str(mapping))

        for img_file in os.listdir(folder_dir):
            img_path = os.path.join(folder_dir, img_file)

            img = Image.open(img_path).convert('L')

            if img.size == (24, 24):
                padded_img = Image.new('L', (28, 28), color=255)
                padded_img.paste(img, (2, 2))
                img = padded_img

            img_array = np.array(img).flatten()   

            X.append(img_array)

            new_mapping = mapping - 33

            if new_mapping == 966:
                new_mapping = 93

            y.append(new_mapping)

    X = np.array(X)
    y = np.array(y)

    return X, y