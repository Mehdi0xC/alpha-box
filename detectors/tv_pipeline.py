import torch
import torchvision

class ObjectDetector:
    def __init__(self, config, labels):
        model = config["model"] 
        threshold = config["threshold"] 
        if model == 'resnet50':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif model == 'resnet101':
            self.model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.labels = labels
        self.threshold = threshold

    def preprocess(self, image):
    
        image = torchvision.transforms.ToTensor()(image)
        image = image.to(self.device)
        image = image.unsqueeze(0)
        return image

    def infer(self, image):
        predictions = self.model(image)
        labels = set()
        for idx, label in enumerate(predictions[0]["labels"]): 
            if predictions[0]["scores"][idx] > self.threshold:
                labels.add(self.labels[label])
        return labels


