#!/usr/bin/env python3
import argparse
import json
import os
import pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Model path to use.")
parser.add_argument("images", nargs="+", type=str, help="Image paths to predict.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

class Dataset:
    class Class:
        def __init__(self, name: str, column: int) -> None:
            self.name = name
            self.column = column
            self.values = []

def main(args: argparse.Namespace) -> None:
    # Use the given number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load the trained model and additional data
    model = tf.keras.models.load_model(args.model, compile=False)
    with open(os.path.join(args.model, "options.json"), "r") as json_file:
        model_args = argparse.Namespace(**json.load(json_file))
    with open(os.path.join(args.model, "classes.pickle"), "rb") as classes_file:
        classes = pickle.load(classes_file)

    # Build pipeline for the dev data
    def load_and_resize(path):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, dtype=tf.float32, channels=3, expand_animations=False)
        image = tf.image.resize(image, [model_args.target_height, model_args.target_width], preserve_aspect_ratio=True)
        return image

    images = tf.data.Dataset.from_tensor_slices(args.images)
    images = images.map(load_and_resize)
    images = images.padded_batch(args.batch_size)

    # Perform prediction
    predictions = model.predict(images)

    if len(classes) == 1:
        predictions = np.expand_dims(predictions, axis=0)

    # Generate the output
    columns = ["image_path"] + [cls.name for cls in classes]
    print(*columns, sep="\t")

    for i in range(len(args.images)):
        values = [args.images[i]]
        for j, cls in enumerate(classes):
            values.append(cls.values[np.argmax(predictions[j][i])])
        print(*values, sep="\t")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
