# EarlyExitNN

This repository contains code for dynamically configurable neural networks with early exit branches. The structure allows for flexible backbone and exit head configurations. The code supports custom backbones as well as pre-trained models from torchvision and timm libraries.

# Project Structure

```
EarlyExitNN-Experiments/
│
├── models/
│   ├── backbone.py
│   ├── exit_head.py
│   ├── common.py
│   └── __init__.py
│
├── scripts/
│   ├── train.py
│   ├── val.py
│   ├── export.py
│   └── infer.py
│
├── configs/
│   ├── model_config.yaml
│   ├── train_config.yaml
│   └── eval_config.yaml
│
├── data/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── utils/
│   ├── dataset.py
│   ├── early_exit.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore
```

# Configuration

The model configuration is specified in the `configs/model_config.yaml` file. Here, you can define the backbone type, library, and layers for both the backbone and exit heads.

# Usage

## Installation

1. Clone the repository

```
git clone https://github.com/yourusername/EarlyExitNN.git
cd EarlyExitNN
```

2. Install the required packages

```
pip install -r requirements.txt
```

## Training

1.	Prepare your dataset and place the images in the data/train/ and data/val/ directories.
2.	Update the configuration file (configs/model_config.yaml) as needed.
3.	Run the training script:
    ```
    python scripts/train.py
    ```

## Evaluation

To evaluate the model, update the configs/eval_config.yaml file and run the evaluation script:

```
python scripts/val.py
```

## Inference

For inference, use the infer.py script:

```
python scripts/infer.py --image path_to_image.jpg
```

## Export

To export the model, use the export.py script:

```
python scripts/export.py --output path_to_output.onnx
```

# Contributing

Contributions are welcome! Please open an issue or submit a pull request.