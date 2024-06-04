import torch.nn as nn
from common import *
import torchvision.models as torchvision_models
import timm


class ModelBackbone(nn.Module):
    def __init__(self, config):
        super(ModelBackbone, self).__init__()
        self.type = config['model']['backbone']['type']
        self.library = config['model']['backbone']['library']
        
        if self.type == 'custom':
            layers = []
            for layer in config['model']['backbone']['layers']:
                name, args = layer[0], layer[1]
                layers.append(globals()[name](*args))
            self.model = nn.Sequential(*layers)
            self.feature_channels = [64, 128, 256, 512]  # Replace with correct sizes for custom backbone
        elif self.library == 'torchvision':
            if self.type == 'resnet18':
                backbone = torchvision_models.resnet18(pretrained=config['pretrained'])
                self.model = nn.Sequential(*list(backbone.children())[:-2])
                self.feature_channels = [64, 128, 256, 512]
            # Add more torchvision backbones as needed
        elif self.library == 'timm':
            backbone = timm.create_model(self.type, pretrained=config['pretrained'], features_only=True)
            self.model = backbone
            self.feature_channels = backbone.feature_info.channels()
            # Alternatively, infer the feature channels if `features_only` is not available

    def forward(self, x):
        if self.type == 'custom':
            outputs = []
            for layer in self.model:
                x = layer(x)
                outputs.append(x)
            return outputs
        else:
            return self.model(x)


if __name__ == "__main__":
    import yaml
    import torch

    with open('configs/custom_cnn.yaml', 'r') as file:
        config = yaml.safe_load(file)
    backbone = ModelBackbone(config)
    print(backbone)
    x = torch.randn(1, 3, 224, 224)
    y = backbone(x)
    for i, yi in enumerate(y):
        print(f'Output {i} shape:', yi.shape)
