import json
from pathlib import Path

def get_labels(config):
    if config["labels"] == "COCO_LABELS":
        # read labels from json file
        with open(Path(Path.cwd(), "detectors/labeler/label_files/COCO_LABELS.json"), "r") as f:
            labels = json.load(f)
        return labels


if __name__ == "__main__":
    get_labels("COCO_LABELS")