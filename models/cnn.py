from typing import Tuple
import torch
import torch.nn as nn


class EarlyExitCNN(nn.Module):
    """
        Custom model with 3 exits and a final classifier.
        Args:
            num_classes (int): Number of classes in the dataset.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Model backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Exit classifiers with dropout
        # Exit-1 classifier (using 64x55x55 output)
        self.exit1 = nn.Sequential(
            nn.Linear(64 * 55 * 55, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Exit-2 classifier (using 128x27x27 output)
        self.exit2 = nn.Sequential(
            nn.Linear(128 * 27 * 27, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Exit-3 classifier (using 512x3x3 output)
        self.exit3 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Final classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing exit-1, exit-2, exit-3 and final classifier outputs.
        """

        x = self.backbone[:6](x)
        exit1_out = self.exit1(x.view(x.size(0), -1))
        
        x = self.backbone[6:10](x)
        exit2_out = self.exit2(x.view(x.size(0), -1))
        
        x = self.backbone[10:](x)
        exit3_out = self.exit3(x.view(x.size(0), -1))
        
        final_out = self.classifier(x.view(x.size(0), -1))
        
        return exit1_out, exit2_out, exit3_out, final_out


if __name__ == "__main__":
    model = EarlyExitCNN()
    x = torch.randn(1, 3, 224, 224)
    exit1_out, exit2_out, exit3_out, final_out = model(x)
    print("Exit-1 output shape:", exit1_out.shape)
    print("Exit-2 output shape:", exit2_out.shape)
    print("Exit-3 output shape:", exit3_out.shape)
    print("Final output shape:", final_out.shape)
