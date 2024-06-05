import torch
import torch.nn as nn


class CustomBackbone(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()

        # Define the feature extractor part of the network with batch normalization
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3, stride=2),
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

        # Define intermediate classifiers for early exits with dropout
        self.exit1 = nn.Sequential(
            nn.Linear(64 * 55 * 55, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.exit2 = nn.Sequential(
            nn.Linear(128 * 27 * 27, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
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

        self.exit_number = 0
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features[:6](x)
        # Exit-1
        exit1_out = self.exit1(x.view(x.size(0), -1))
        x = self.features[6:10](x)
        # Exit-2
        exit2_out = self.exit2(x.view(x.size(0), -1))
        x = self.features[10:](x)
        # Exit-3
        exit3_out = self.exit3(x.view(x.size(0), -1))
        # Final exit
        final_out = self.classifier(x.view(x.size(0), -1))
        
        return exit1_out, exit2_out, exit3_out, final_out


if __name__ == "__main__":
    model = CustomBackbone(in_channels=3, num_classes=2)
    print("Model architecture: ", model)

    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out[0].shape}, {out[1].shape}, {out[2].shape}, {out[3].shape}")
