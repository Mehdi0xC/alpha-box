from transformers import DetrForObjectDetection, DetrFeatureExtractor
import torchvision
class ObjectDetector:
    def __init__(self, config, labels):
        model = config["model"] 
        threshold = config["threshold"] 
        if model == 'resnet101':  
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
            self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
        else:
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

        self.threshold = threshold
        self.labels = labels

    def preprocess(self, image):
        image = torchvision.transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        return image

    def infer(self, image):
        outputs = self.model(image)
        label = self.feature_extractor.post_process_object_detection(outputs)
        results = {}
        results["label"] = None
        results["score"] = None
        results["bbox"] = None
        if label[0]["scores"].numel() and label[0]["scores"][0] > self.threshold:
            label_idx = label[0]["labels"][0].item()    
            results["label"] = self.labels[label_idx]
            results["score"] = label[0]["scores"][0].item()
            results["bbox"] = label[0]["boxes"][0].tolist()
        return results

# [{'scores': tensor([], grad_fn=<IndexBackward0>), 'labels': tensor([], dtype=torch.int64), 'boxes': tensor([], size=(0, 4), grad_fn=<IndexBackward0>)}]
