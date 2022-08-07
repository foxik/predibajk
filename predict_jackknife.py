#!/usr/bin/env python3
import argparse
import json
import os
import pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from model_train import Dataset, Model

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Model path to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load the data and the trained model
    with open(os.path.join(args.model, "options.json"), "r") as json_file:
        model_args = argparse.Namespace(**json.load(json_file))
    data = Dataset(model_args)
    model = Model(model_args, data.classes)
    model.load_weights(args.model)

    # Build pipeline for the dev data
    dev = model.pipeline(data.dev, model_args, training=False)

    # Perform prediction
    predictions = model.predict(dev)

    if len(data.classes) == 1:
        predictions = np.expand_dims(predictions, axis=0)

    # Generate the output
    columns = ["image_path"]
    for cls in data.classes:
        columns.extend([cls.name, cls.name + "_predicted"])
    print(*columns, sep="\t")

    for i in range(len(data.dev["image_path"])):
        values = [data.dev["image_path"][i]]
        for j, cls in enumerate(data.classes):
            values.extend([cls.values[data.dev["classes"][i, j]], cls.values[np.argmax(predictions[j][i])]])
        print(*values, sep="\t")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
