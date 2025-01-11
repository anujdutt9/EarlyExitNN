from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights


class EarlyExitResNet(nn.Module):
    """
        Custom model with ResNet18 backbone, 3 exits and a final classifier.
        Args:
            num_classes (int): Number of classes in the dataset.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        
        # Load pretrained ResNet-18 and extract its layers
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Model backbone
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Exit classifiers with dropout
        # Exit-1 classifier (using resnet.layer1 output)
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.layer1[-1].conv1.out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Exit-2 classifier (using resnet.layer2 output)
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.layer2[-1].conv1.out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Exit-3 classifier (using resnet.layer3 output)
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.layer3[-1].conv1.out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Final classifier (using resnet.layer4 output)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.layer4[-1].conv1.out_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the output of each exit.
        """

        x = self.backbone[0:5](x)
        exit1_out = self.exit1(x)
        
        x = self.backbone[5:6](x)
        exit2_out = self.exit2(x)
        
        x = self.backbone[6:7](x)
        exit3_out = self.exit3(x)

        x = self.backbone[7:](x)
        final_out = self.classifier(x)
        
        return exit1_out, exit2_out, exit3_out, final_out


if __name__ == "__main__":
    # Create model
    model = EarlyExitResNet(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    exit1_out, exit2_out, exit3_out, final_out = model(x)
    print("Exit-1 output shape:", exit1_out.shape)
    print("Exit-2 output shape:", exit2_out.shape)
    print("Exit-3 output shape:", exit3_out.shape)
    print("Final output shape:", final_out.shape)
