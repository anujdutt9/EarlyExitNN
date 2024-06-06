import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBackbone(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(CustomBackbone, self).__init__()

        # Feature extractor part of the network with batch normalization
        self.backbone = nn.Sequential(
            self._create_backbone_layer(in_channels, 32, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            self._create_backbone_layer(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._create_backbone_layer(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._create_backbone_layer(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._create_backbone_layer(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Intermediate classifiers for early exits with dropout
        self.exit1 = self._create_exit_layer(64 * 55 * 55, num_classes)
        self.exit2 = self._create_exit_layer(128 * 27 * 27, num_classes)
        self.exit3 = self._create_exit_layer(512 * 3 * 3, num_classes)

        # Final classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self._initialize_weights()

    def _create_backbone_layer(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _create_exit_layer(self, input_dim: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.backbone[:3](x)
        print(x.shape)
        exit1_out = self.exit1(x.view(x.size(0), -1))
        
        x = self.backbone[3:4](x)
        exit2_out = self.exit2(x.view(x.size(0), -1))
        
        x = self.backbone[4:](x)
        exit3_out = self.exit3(x.view(x.size(0), -1))
        
        final_out = self.classifier(x.view(x.size(0), -1))
        
        return exit1_out, exit2_out, exit3_out, final_out

if __name__ == "__main__":
    model = CustomBackbone(in_channels=3, num_classes=2)
    print("Model architecture: ", model)

    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shapes: {out[0].shape}, {out[1].shape}, {out[2].shape}, {out[3].shape}")