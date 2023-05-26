import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf


def predict(images_path, csv_path):
    loaded_model = tf.keras.models.load_model("model.h5")
    loaded_model.summary()

    dicom_list = os.listdir(images_path)
    labels = {"SOPInstanceUID": [], "Label": []}
    for dcm in dicom_list[:50]:
        if "dcm" == dcm.split(".")[-1]:
            ds = pydicom.dcmread(os.path.join(images_path, dcm))
            img = ds.pixel_array
            image = cv2.resize(img, (550, 1100)) / 255.0
            lbl = loaded_model.predict(image.reshape((1, 550, 1100)))
            label = np.where(lbl > 0.5, 1, 0)[0][0]
            labels["SOPInstanceUID"].append(dcm[:-4])
            labels["Label"].append(label)

    df = pd.DataFrame(labels)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="path to folder containing test images")
    parser.add_argument("--output", help="path to final CSV output")
    args = parser.parse_args()
    predict(args.inputs, args.output)
