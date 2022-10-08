# EEG-Classification

EEG Classification of [https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection) data

## Data

[https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection/data](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection/data)

Extract `train.zip` to `data/train` folder

## Dependencies

```bash
pip install -r requirements.txt
```

## Training

Available models: basic, inception, resnet

```bash
python train_{model}.py
```

eg:

```bash
python train_inception.py
```

ML Models are still WIP

## Testing on validation set

```bash
python validate.py
```

Change the model and state file as required in `validate.py`

## Credits

Basic Net and Loaders: https://www.kaggle.com/code/banggiangle/cnn-eeg-pytorch