from detectors.detector import get_detector
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import copy

# read yaml config file
with open(Path("configs/prebox_detector.yaml"), "r") as f:
    config = yaml.safe_load(f)

model = get_detector(config)
output_path = Path(config["path"]["output"])
input_path = Path(config["path"]["input"])
# create cv2 video writer

width = config["frame_size"]["width"]
height = config["frame_size"]["height"]
fsize = (width, height)
frate = config["frame_rate"]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, frate, fsize)
pre = None

for path in tqdm(sorted(Path.glob(input_path, "*.jpg"))):
    image = Image.open(path)
    image_to_write = np.array(image)
    image = model.preprocess(image)
    outputs = model.infer(image)
    if outputs["bbox"] is not None:
        bbox = outputs["bbox"]
        # rescale bounding boxes
        bbox[0] = int(bbox[0] * width)
        bbox[1] = int(bbox[1] * height)
        bbox[2] = int(bbox[2] * width)
        bbox[3] = int(bbox[3] * height)
        x_center = int((bbox[0] + bbox[2])/2)
        # cv2.rectangle(image_to_write, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # draw a vertical line passing from the center of the bbox 
        # image_to_write = cv2.line(image_to_write, (int(x_center), 0), (int(x_center), 256), (0, 0, 255), thickness=1, lineType = cv2.LINE_AA)
    if pre is not None:
        # temp = image_to_write[:, x_center-128:x_center+128, :]
        # temp = copy.deepcopy(image_to_write)
        image_to_write[:, x_center-128:x_center+128, :] = pre[:, x_center-128:x_center+128, :]
        # fixed = ILVR(pre[:, x_center-128:x_center+128, :])
        # image_to_write[:, x_center-128:x_center+128, :] = fixed

        pre = copy.deepcopy(image_to_write)
    else:
        pre = copy.deepcopy(image_to_write)
    out.write(image_to_write)

# odict_keys(['logits', 'pred_boxes', 'last_hidden_state', 'encoder_last_hidden_state'])
# image = Image.open("horse.jpg")
# print(model.inference(image))