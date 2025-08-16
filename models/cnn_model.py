import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class DiabeticRetinopathyCNN(nn.Module):
    """
    Custom CNN for Diabetic Retinopathy Classification
    """
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(DiabeticRetinopathyCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ResNetDR(nn.Module):
    """
    ResNet-based model for Diabetic Retinopathy Classification
    """
    def __init__(self, num_classes=5, pretrained=True, model_name='resnet50'):
        super(ResNetDR, self).__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output

class EfficientNetDR(nn.Module):
    """
    EfficientNet-based model for Diabetic Retinopathy Classification
    """
    def __init__(self, num_classes=5, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetDR, self).__init__()
        
        try:
            # Try to import efficientnet
            from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
            
            if model_name == 'efficientnet_b0':
                self.backbone = efficientnet_b0(pretrained=pretrained)
                num_features = 1280
            elif model_name == 'efficientnet_b1':
                self.backbone = efficientnet_b1(pretrained=pretrained)
                num_features = 1280
            elif model_name == 'efficientnet_b2':
                self.backbone = efficientnet_b2(pretrained=pretrained)
                num_features = 1408
            else:
                raise ValueError(f"Unsupported EfficientNet model: {model_name}")
            
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
            
        except ImportError:
            print("EfficientNet not available, falling back to ResNet50")
            self.backbone = ResNetDR(num_classes=num_classes, pretrained=pretrained)
    
    def forward(self, x):
        return self.backbone(x)

class AttentionBlock(nn.Module):
    """
    Attention mechanism for focusing on important regions
    """
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention

class AttentionCNN(nn.Module):
    """
    CNN with attention mechanism for Diabetic Retinopathy
    """
    def __init__(self, num_classes=5):
        super(AttentionCNN, self).__init__()
        
        # Base CNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Attention blocks
        self.attention1 = AttentionBlock(64, 64)
        self.attention2 = AttentionBlock(128, 128)
        self.attention3 = AttentionBlock(256, 256)
        
        # Pooling and classification
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Layer 1 with attention
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        x = self.pool(x)
        
        # Layer 2 with attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        x = self.pool(x)
        
        # Layer 3 with attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        x = self.pool(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def create_model(model_type='resnet50', num_classes=5, pretrained=True):
    """
    Factory function to create different model architectures
    """
    if model_type == 'custom_cnn':
        return DiabeticRetinopathyCNN(num_classes=num_classes)
    elif model_type.startswith('resnet'):
        return ResNetDR(num_classes=num_classes, pretrained=pretrained, model_name=model_type)
    elif model_type.startswith('efficientnet'):
        return EfficientNetDR(num_classes=num_classes, model_name=model_type, pretrained=pretrained)
    elif model_type == 'attention_cnn':
        return AttentionCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    models_to_test = ['custom_cnn', 'resnet50', 'attention_cnn']
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name, num_classes=5)
            params = count_parameters(model)
            print(f"{model_name}: {params:,} trainable parameters")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print("✓ Model created successfully\n")
            
        except Exception as e:
            print(f"✗ Error creating {model_name}: {e}\n")