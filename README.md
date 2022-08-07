# PrediBajk

The repository contains the following scripts:

- `download_images.sh`: given an input `tsv` file with URL in the first column,
  download the images, resize them to height 480px, save them as `jpg`s, and
  print the output `tsv` with the image path prepended as the first column.

- `model_train.py`: train a model using an EfficientNetV2 backbone. The trained
  is saved as Keras model in `SavedModel` format. The script has the following
  options:
  - `--augment` (default `flip,rotate,scale,colors`): augmentation operations to
    apply to the training data
  - `--batch_size` (default `8`): batch size to use; 8 works on a 16GB GPU with
    a small variant of EfficientNetV2, 4 works on a 24GB GPU with a medium
    EfficientNetV2
  - `--data_classes` (default `all`): which classes from the input data file
    to predict, or `all` to predict all of them
  - `--data_folds` (default `0`): number of folds to use during cross-validation;
    value of 0 indicates not to use cross-validation
  - `--data_fold` (default `0`): zero-based index of a fold to use during cross-validation
  - `--data_path` (default `bikes.tsv`): input data file in `tsv` format
  - `--dropout` (default `0.0`): dropout rate to use
  - `--efficientnetv2_size` (default `s`): EfficientNetV2 size (`s` or `m`)
  - `--output` (default `model`): path where the trained model should be saved
  - `--schedule` (default `frozen:30:1e-3,finetune:30:1e-4`): training schedule to
    use
  - `--seed` (default `42`): random seed to use
  - `--threads` (default `4`): CPU threads to use (for augmentation and other CPU
    computation); 0 indicates all physical cores

  The script has been developed using `tensorflow==2.8.0`, `tensorflow-addons==0.16.1`,
  and `tensorflow-hub==0.12.0`, but newer (consistent) versions are expected to
  work too.

- `predict.py`: predict the classes for the given images. The first argument
  is the directory with the model, all remaining arguments are paths to the
  filed to predict. Furthermore, the following options are supported:
  - `--batch_size` (default `8`): the batch size to use during prediction
  - `--threads` (default `4`): CPU threads to use; 0 indicates all physical cores

  The output is a `tsv` file, the first column `image_path` is the path to the
  file being predicted, and other columns are the predicted classes.

  Only the `tensorflow` package is required for prediction.

- `predict_jackknife.py`: when a model (given as the first argument) was trained
  via cross-validation, perform the prediction on the development fold (the one
  not used during training). Output a `tsv` file with both gold and predicted
  classes for each file in the development fold.

- `visualize.py`: an internal script used to visualize the cross-validation results.
  The input (given as the first argument) is a `tsv` file where columns from the
  third are expected to be pairs of gold and predicted values.
