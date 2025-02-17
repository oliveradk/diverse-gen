import torch 
from torch import nn
from torchvision import transforms



def get_model(model_name: str):
    model_transform = None
    pad_sides = False
    tokenizer = None
    if model_name == "Resnet50":
        from torchvision import models
        from torchvision.models.resnet import ResNet50_Weights
        resnet_builder = lambda: models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)    
        model_builder = lambda: torch.nn.Sequential(*list(resnet_builder().children())[:-1])
        resnet_50_transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
        model_transform = transforms.Compose([
            transforms.Resize(resnet_50_transforms.resize_size * 2, interpolation=resnet_50_transforms.interpolation),
            transforms.CenterCrop(resnet_50_transforms.crop_size),
            transforms.Normalize(mean=resnet_50_transforms.mean, std=resnet_50_transforms.std)
        ])
        pad_sides = True
        feature_dim = 2048
    elif model_name == "ClipViT":
        import clip 
        preprocess = clip.clip._transform(224)
        clip_builder = lambda: clip.load('ViT-B/32', device='cpu')[0]
        model_builder = lambda: clip_builder().visual
        model_transform = transforms.Compose([
            preprocess.transforms[0],
            preprocess.transforms[1],
            preprocess.transforms[4]
        ])
        feature_dim = 512
        pad_sides = True
    elif model_name == "bert":
        from transformers import BertModel, BertTokenizer
        from diverse_gen.models.hf_wrapper import HFWrapper
        bert_builder = lambda: BertModel.from_pretrained('bert-base-uncased')
        model_builder = lambda: HFWrapper(bert_builder())
        feature_dim = 768
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == "toy_model":
        model_builder = lambda: nn.Sequential(
            nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU()
        )
        feature_dim = 40
    elif model_name == "LeNet":
        from diverse_gen.models.lenet import LeNet
        from functools import partial
        model_builder = lambda: partial(LeNet, num_classes=1, dropout_p=0.0)
        feature_dim = 256
    else: 
        raise ValueError(f"Model {model_name} not supported")
    
    model_dict = {
        "model_builder": model_builder, 
        "model_transform": model_transform, 
        "feature_dim": feature_dim, 
        "pad_sides": pad_sides, 
        "tokenizer": tokenizer
    }
    return model_dict