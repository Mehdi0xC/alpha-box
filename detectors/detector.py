
# define getter and setter for the detector
from detectors.labeler.label_reader import get_labels


def get_detector(config):
    labels = get_labels(config)
    if config["pipeline"] == "tv":
        from detectors.tv_pipeline import ObjectDetector
        return ObjectDetector(config, labels)
    elif config["pipeline"] == "detr":
        from detectors.detr_pipeline import ObjectDetector
        return ObjectDetector(config, labels)