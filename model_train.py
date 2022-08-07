#!/usr/bin/env python3
import argparse
import json
import os
import pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

parser = argparse.ArgumentParser()
parser.add_argument("--augment", default="flip,rotate,scale,colors", type=str, help="Augment data.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--data_classes", default="all", type=str, help="Which classes to predict.")
parser.add_argument("--data_fold", default=0, type=int, help="Index of the fold to use.")
parser.add_argument("--data_folds", default=0, type=int, help="Number of folds to use.")
parser.add_argument("--data_path", default="bikes.tsv", type=str, help="Data file path.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")
parser.add_argument("--efficientnetv2_size", default="s", type=str, help="EfficientNetV2 size.")
parser.add_argument("--output", default="model", type=str, help="Output model path.")
parser.add_argument("--schedule", default="frozen:30:1e-3,finetune:30:1e-4", type=str, help="Training schedule")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

class Dataset:
    class Class:
        def __init__(self, name: str, column: int) -> None:
            self.name = name
            self.column = column
            self.values = []

    def __init__(self, args: argparse.Namespace) -> None:
        with open(args.data_path, "r", encoding="utf-8-sig") as data_file:
            # Load classed index
            columns = data_file.readline().rstrip("\r\n").split("\t")
            image_path_column = columns.index("image_path")
            self.classes = []
            for i, column in enumerate(columns):
                if column in ["image_path", "main_photo_url", "image"]:
                    continue
                if column in args.data_classes or "all" in args.data_classes:
                    self.classes.append(self.Class(column, i))

            # Load data
            self.data = {"image_path": [], "classes": []}
            for line in data_file:
                columns = line.rstrip("\r\n").split("\t")
                self.data["image_path"].append(columns[image_path_column])
                classes = []
                for cls in self.classes:
                    try:
                        index = cls.values.index(columns[cls.column])
                    except ValueError:
                        index = len(cls.values)
                        cls.values.append(columns[cls.column])
                    classes.append(index)
                self.data["classes"].append(classes)
            self.data = {key: np.array(value) for key, value in self.data.items()}

        # Create data split
        if args.data_folds == 0:
            self.train = self.data
            self.dev = None
        else:
            permutation = np.random.RandomState(42).permutation(len(self.data["image_path"]))
            folds = np.array_split(permutation, args.data_folds)
            dev_indices = folds.pop(args.data_fold)
            train_indices = np.concatenate(folds, axis=0)
            self.train = {key: value[train_indices] for key, value in self.data.items()}
            self.dev = {key: value[dev_indices] for key, value in self.data.items()}

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, classes: list[Dataset.Class]) -> None:
        # Create the model
        backbone = hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_{}/feature_vector/2".format(
                args.efficientnetv2_size), trainable=True)

        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        features = backbone(inputs)
        outputs = [
            tf.keras.layers.Dense(len(cls.values), activation=tf.nn.softmax, name=cls.name)(
                tf.keras.layers.Dropout(args.dropout)(features))
            for cls in classes
        ]
        super().__init__(inputs=inputs, outputs=outputs)

        def set_backbone_trainable(trainable):
            backbone.trainable = trainable
        self.set_backbone_trainable = set_backbone_trainable

    def pipeline(self, data: dict[str, np.ndarray], args: argparse.Namespace, training: bool = False) -> tf.data.Dataset:
        data = tf.data.Dataset.from_tensor_slices(data)
        if training:
            data = data.shuffle(len(data), seed=args.seed)
        def process_example(example):
            return (
                tf.io.decode_image(tf.io.read_file(example["image_path"]), dtype=tf.float32, channels=3, expand_animations=False),
                tuple(tf.unstack(example["classes"], axis=-1)),
            )
        data = data.map(process_example, num_parallel_calls=tf.data.AUTOTUNE)
        if training and args.augment:
            generator = tf.random.Generator.from_seed(args.seed)
            def augment_image(image, label):
                image_size = tf.shape(image)[:2]
                if "flip" in args.augment and generator.uniform([]) >= 0.5:
                    image = tf.image.flip_left_right(image)
                image = tf.image.resize_with_crop_or_pad(image, image_size[0] + 40, image_size[1] + 40)
                if "rotate" in args.augment:
                    image = tfa.image.rotate(
                        image, generator.uniform([], minval=np.deg2rad(-15), maxval=np.deg2rad(15)), interpolation="bilinear")
                if "scale" in args.augment:
                    image = tf.image.resize(
                        image, [generator.uniform([], minval=image_size[0], maxval=image_size[0] + 80, dtype=tf.int32),
                                generator.uniform([], minval=image_size[1], maxval=image_size[1] + 80, dtype=tf.int32)]
                    )
                image = tf.image.crop_to_bounding_box(
                    image, target_height=image_size[0], target_width=image_size[1],
                    offset_height=generator.uniform([], maxval=tf.shape(image)[0] - image_size[0] + 1, dtype=tf.int32),
                    offset_width=generator.uniform([], maxval=tf.shape(image)[1] - image_size[1] + 1, dtype=tf.int32),
                )
                image = tf.image.resize(image, [args.target_height, args.target_width], preserve_aspect_ratio=True)
                if "colors" in args.augment:
                    image = tf.image.random_contrast(image, 0.8, 1.2)
                    image = tf.image.random_saturation(image, 0.8, 1.2)
                    image = tf.image.random_brightness(image, 0.2)
                return image, label
        else:
            def augment_image(image, label):
                image = tf.image.resize(image, [args.target_height, args.target_width], preserve_aspect_ratio=True)
                return image, label
        data = data.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.padded_batch(args.batch_size)
        data = data.prefetch(tf.data.AUTOTUNE)
        return data


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Post-process arguments
    args.data_classes = args.data_classes.split(",")
    args.schedule = [(mode, int(epochs), float(lr)) for stage in args.schedule.split(",") for mode, epochs, lr in [stage.split(":")]]
    args.target_height = 384 if args.efficientnetv2_size == "s" else 480
    args.target_width = 5 * args.target_height // 2

    # Load the data
    data = Dataset(args)

    # Create the model
    model = Model(args, data.classes)

    # Save the value mappings
    os.makedirs(args.output)
    with open(os.path.join(args.output, "options.json"), "w") as json_file:
        json.dump(vars(args), json_file, sort_keys=True, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "classes.pickle"), "wb") as classes_file:
        pickle.dump(data.classes, classes_file, protocol=3)
    logs_file = open(os.path.join(args.output, "logs.txt"), "w")

    # Build the input pipelines
    train = model.pipeline(data.train, args, training=True)
    dev = data.dev and model.pipeline(data.dev, args, training=False)

    # Training
    total_epochs = 0
    for mode, epochs, lr in args.schedule:
        if mode == "frozen":
            model.set_backbone_trainable(False)
        elif mode == "finetune":
            model.set_backbone_trainable(True)
        else:
            raise ValueError("Unknown mode {}".format(mode))
        lr = tf.optimizers.schedules.CosineDecay(lr, epochs * len(train))
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=[tf.losses.SparseCategoricalCrossentropy() for _ in data.classes],
            metrics=[[tf.metrics.SparseCategoricalAccuracy("acc")] for cls in data.classes],
        )
        logs = model.fit(train, epochs=total_epochs + epochs, initial_epoch=total_epochs, validation_data=dev)
        total_epochs += epochs
        print("Epoch={}".format(total_epochs),
              *["{}={}".format(name, values[-1]) for name, values in logs.history.items() if name.startswith("val_") and name.endswith("_acc")],
              file=logs_file, flush=True)

    model.set_backbone_trainable(False)
    model.compile()
    model.save(args.output, save_format="tf", include_optimizer=False, save_traces=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
