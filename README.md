# MySunSign | Machine Learning Repository

## Overview

This repository is for storing jupyter notebooks that are used for developing and experimenting machine learning models that used in our MySunSign app for TTS and Sign Language Detection feature.

Below is the folder structure of the repository:

```
Machine-Learning
│
├── Experiment
│   ├── Experiment_Crop_Hand_Saliency_Map.ipynb
│   ├── Experiment_Image_Recognition.ipynb
│   ├── Experiment_Model_Evaluation.ipynb
│   ├── Experiment_Resnet50_modelv1v5.ipynb
│   ├── Experiment_tts_model.ipynb
│   └── finetune_mms_ind.json
│
├── Main
│   ├──  Image_Recognition.ipynb
│   └──  TTS_inference.ipynb
│
├── Utils
│   └── split_data_image.ipynb
└── README.md
```

The `Main` folder contains notebooks with end-to-end code for model training, validation, and inference, which are deployed in the MySunSign app.

The `Experiment` folder contains notebooks for experiments conducted during model development.

The `Utils` folder includes notebooks for utility tasks, such as data splitting.

## Datasets

### Image Classification

We used these datasets to develop the Image Classification model:

- [ASL(American Sign Language) Alphabet Dataset Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) -> Training and validation set
- [ASL Alphabet Test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test?select=W) -> Test set
- [Authors Data Collection](https://drive.google.com/drive/folders/1C8Mdt693a2gOh02DkB6v5cn_w879ER3x?usp=sharing) -> Test set

### TTS

We used this dataset to fine tune the TTS model:

- [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia)

## Models

### Image Classification

For Sign Language Classification task, we tried 6 iterations to develop this model:

1. Base model: CNN + Linear layer (Custom architecture, trained from scratch)
2. v1: Resnet50 + Linear layer + Softmax (Transfer learning, ResNet50 weights frozen, no preprocess, only used zoomed in hands images)
3. v2: Resnet50 + Linear layer + Softmax (Transfer learning, ResNet50 weights frozen, preprocess, mixed data of zoomed in and zoomed out hands images)
4. v3: Resnet50 + Linear layer + Softmax (Transfer learning, Unfroze several blocks in the later layers of Resnet50, preprocess (crop hands + augmentation), mixed data of zoomed in and zoomed out hands images, 1 epoch)
5. v4: Resnet50 + Linear layer + Softmax (Transfer learning, Unfroze several blocks in the later layers of Resnet50, preprocess (crop hands + augmentation), mixed data of zoomed in and zoomed out hands images, 2 epochs)
6. v5: Resnet50 + Linear layer + Softmax (Transfer learning, ResNet50 weights frozen, preprocess (crop hands + augmentation), mixed data of zoomed in and zoomed out hands images, 5 epochs)

We used v4 model as the final model, because it scored the highest on the test set (97.2%).

You can access the models [here](https://drive.google.com/drive/folders/1_AdyiJEpuFgQhDOjNsfsBfCBzRs_8b_U?usp=sharing).

### TTS

For TTS model, we used VITS based architecture pretrained model named [mms-tts-ind](https://huggingface.co/facebook/mms-tts-ind).
Here is the finetuned model [mms-tts-ind-finetuned](https://huggingface.co/fadhilamri/tts-mms-ind-finetuned).

## Notebook Details

### Image Classification

`Experiment_Crop_Hand_Saliency_Map.ipynb`: Experiments to check crop hand preprocessing using `mediapipe` library and check saliency map of models after training/finetuning.

`Experiment_Image_Recognition.ipynb`: Complete preprocessing and training/e experiments to develop

`Experiment_Model_Evaluation.ipynb`: Model evaluation on test sets.

`Experiment_Resnet50_modelv1v5.ipynb`: End-to-end experiments for v1 and v5 models (from preprocessing to evaluation).

`Image_Recognition.ipynb`: End-to-end pipeline for Sign Language Detection task + deployed final inference code.

`split_data_image.ipynb`: Split initial data to training and validation set.

### TTS

`Experiment_tts_model.ipynb`: Experiment to fine tune TTS model utilizing open source framework for VITS architecture.

`finetune_mms_ind.json`: Configuration file for finetuning VITS model.

## Authors

| Name                   | Bangkit ID   |
| ---------------------- | ------------ |
| Muhammad Fadhil Amri   | M002B4KY3368 |
| Nigel Sahl             | M002B4KY2799 |
| Hanif Muhammad Zhafran | M002B4KY1709 |
