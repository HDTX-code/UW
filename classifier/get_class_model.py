import torch
import torchvision


def get_class_model(num_classes, backbone='resnet50', pretrained=False):
    model = getattr(torchvision.models, backbone)(pretrained=pretrained)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    return model