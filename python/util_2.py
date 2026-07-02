import os
import shutil
import json
import random
import subprocess
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt


# exiftool executable. Override with the EXIFTOOL env var (e.g. "exiftool" on Linux);
# otherwise fall back to the bundled Windows exe (project_root/exiftool-13.59_64/exiftool.exe)
EXIFTOOL_PATH = os.environ.get("EXIFTOOL") or os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "exiftool-13.59_64", "exiftool.exe")
)


def read_metadata_exiftool(image_path):
    """
    Read the EXIF / DJI maker-note fields needed for geolocation using exiftool.
    Returns a dict of the requested tags (numeric values, thanks to -n).
    """
    result = subprocess.run(
        [
            EXIFTOOL_PATH,
            "-j",   # JSON output
            "-n",   # numeric values (decimal GPS, raw refs, degrees, etc.)
            "-GPSLatitude", "-GPSLatitudeRef",
            "-GPSLongitude", "-GPSLongitudeRef",
            "-GPSAltitude",
            "-RelativeAltitude",
            "-FocalLength",
            "-ImageWidth", "-ImageHeight",
            "-GimbalYawDegree", "-GimbalPitchDegree", "-GimbalRollDegree",
            "-FlightYawDegree", "-FlightPitchDegree", "-FlightRollDegree",
            image_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed for {image_path}: {result.stderr.strip()}")

    data = json.loads(result.stdout) if result.stdout.strip() else []
    return data[0] if data else {}


def write_metadata_csv(image_paths, output_file):
    """
    Run exiftool on every given image and write the metadata of all of them
    into a single CSV file (one row per image, tags as columns).
    """
    if not image_paths:
        return None

    result = subprocess.run(
        [EXIFTOOL_PATH, "-csv", *image_paths],
        capture_output=True,
        text=True,
    )

    # exiftool emits CRLF rows; newline="" keeps them clean across platforms
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write(result.stdout)

    return output_file


def get_model(num_classes, pretrain):
    if pretrain:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_model_mask(num_classes, pretrain, min_size=800, max_size=1333):
    if pretrain:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size
        )
    else:
        print("No pretraining")
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size
        )
    # --- Replace box predictor ---
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # --- Replace mask predictor ---
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
class CustomImageDataset(Dataset):
    def __init__(self, data, root_dir, to_tensor=True, max_size=1024):
        """
        Args:
            data (list): List of dictionaries, each containing 'image_id' and 'annotations'.
            root_dir (str): Folder where images are stored.
            to_tensor (bool): Whether to convert images to tensor.
            max_size (int): Maximum size of the longer edge of the image.
        """
        self.data = data
        self.root_dir = root_dir
        self.to_tensor = to_tensor
        self.max_size = max_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.root_dir, f"{item['image_id']}.jpg").replace("\\", "/")
        image = Image.open(img_path).convert("RGB")

        # --- Resize image while keeping aspect ratio ---
        w, h = image.size
        scale = self.max_size / max(w, h) if max(w, h) > self.max_size else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)

        image_tensor = TF.to_tensor(image) if self.to_tensor else image

        boxes, labels, masks = [], [], []

        for ann in item['annotations']:
            x, y, w_box, h_box = ann['bbox']
            # Scale boxes
            x, y, w_box, h_box = x*scale, y*scale, w_box*scale, h_box*scale
            if w_box > 0 and h_box > 0:
                boxes.append([x, y, x + w_box, y + h_box])
                labels.append(max(1, ann.get('category_id', 1)))

                # Generate and resize mask
                mask = draw_binary_map(
                    ann.get('all_points_x', [x, x+w_box, x+w_box, x]),
                    ann.get('all_points_y', [y, y, y+h_box, y+h_box]),
                    image_size=(int(image.width / scale), int(image.height / scale))  # original mask size
                )
                mask = Image.fromarray(mask).resize((new_w, new_h), resample=Image.NEAREST)
                masks.append(np.array(mask, dtype=np.uint8))
        masks_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8) if masks else torch.zeros((0, new_h, new_w), dtype=torch.uint8)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': masks_tensor
        }

        return image_tensor, target



def get_class_name(class_id):
    classes={0: "Background", 1: "acacia"}
    return classes.get(class_id)
def compute_area(box):
    x_min, y_min, x_max, y_max = box
    return max(0, x_max - x_min) * max(0, y_max - y_min)

def compute_overlap_area(pred_box, gt_box):
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[0] + gt_box[2])
    yB = min(pred_box[3], gt_box[1] + gt_box[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    return inter_area
        

def prepare_image(image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)


import torch
import numpy as np
def evaluate_iou(prediction, target, score_threshold=0.5, pixel_threshold=0.5, iou_threshold=0.5, eps=1e-6):
    """
    Computes IoU between predicted masks and ground truth masks with one-to-one matching.
    Ensures each ground truth mask can only be matched to one prediction (the highest IoU).

    Args:
        prediction (dict): model output for one image, contains 'masks' and 'scores'
        target (dict): ground truth for one image, contains 'masks'
        score_threshold (float): minimum prediction score to consider
        pixel_threshold (float): threshold to binarize predicted masks
        iou_threshold (float): minimum IoU for a valid match
        eps (float): small value to avoid division by zero

    Returns:
        total_iou (float): sum of IoUs for all matched predictions
        mean_iou (float): average IoU across all GTs (unmatched count as 0)
        count (int): total number of GT masks
    """
    import numpy as np

    # --- Extract masks ---
    pred_masks = prediction.get('masks')
    scores = prediction.get('scores')
    gt_masks = target.get('masks')

    if pred_masks is None or gt_masks is None:
        return 0.0, 0.0, 0

    pred_masks = pred_masks.cpu().numpy()  # [N,1,H,W]
    gt_masks = gt_masks.cpu().numpy()      # [M,H,W]

    # --- Filter predictions by score ---
    if scores is not None:
        scores = scores.cpu().numpy()
        keep_idx = scores >= score_threshold
        pred_masks = pred_masks[keep_idx]

    total_iou = 0.0
    total_matched = 0
    total_unmatched = 0
    matched_gt = {}  # {gt_index: (iou_value, pred_index)}

    # --- Compare each prediction with each GT ---
    for p_idx, pred_mask in enumerate(pred_masks):
        pred_binary = (pred_mask[0] > pixel_threshold).astype(np.uint8)
        match=False
        for g_idx, gt_mask in enumerate(gt_masks):
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            intersection_mask = np.logical_and(pred_binary, gt_binary)
            union_mask = np.logical_or(pred_binary, gt_binary)
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()

            iou = intersection / (union + eps)
            # --- Only consider valid matches ---
            if iou >= iou_threshold:
                if g_idx in matched_gt:
                    if iou > matched_gt[g_idx][0]:
                        matched_gt[g_idx] = (iou, p_idx)
                else:
                    matched_gt[g_idx] = (iou, p_idx)
                    total_matched+=1
                    match=True

        if not match:
            total_unmatched+=1
    # --- Sum all IoUs from matched GTs ---
    total_iou = sum([v[0] for v in matched_gt.values()])
    mean_iou = total_iou / len(pred_masks) if len(pred_masks) > 0 else 0.0
    fn=len(gt_masks)-total_matched
    return total_iou, mean_iou, total_matched, fn, total_unmatched


# def evaluate_iou(prediction, target, score_threshold=0.5, pixel_threshold=0.5, iou_threshold=0.5, eps=1e-6):
#     """
#     Computes IoU between predicted masks and ground truth masks.

#     Args:
#         prediction (dict): model output for one image, contains 'masks' and 'scores'
#         target (dict): ground truth for one image, contains 'masks'
#         score_threshold (float): minimum prediction score to consider
#         pixel_threshold (float): threshold to binarize predicted masks
#         eps (float): small value to avoid division by zero

#     Returns:
#         total_iou (float): sum of IoUs for all predicted masks + unmatched GT
#         mean_iou (float): mean IoU for all predictions (unmatched predictions count as 0)
#         count (int): total number of masks considered (predictions + unmatched GT)
#     """
#     # --- Extract masks ---
#     pred_masks = prediction['masks'].cpu() if 'masks' in prediction else None
#     scores = prediction['scores'].cpu() if 'scores' in prediction else None
#     gt_masks = target['masks'].cpu() if 'masks' in target else None

#     if pred_masks is None or gt_masks is None:
#         return 0.0, 0.0, 0

#     # --- Filter predictions by score ---
#     if scores is not None:
#         keep_idx = scores >= score_threshold  # producing boolean tensor e.g. torch.tensor([False, True, True])
#         pred_masks = pred_masks[keep_idx]     # only keeps the masks that are 'True' from the boolean tensor

#     pred_masks = pred_masks.numpy()  # shape: [N,1,H,W]
#     gt_masks = gt_masks.numpy()      # shape: [M,H,W]

#     total_iou = 0.0
#     count = 0

#     # --- Match each predicted mask to best GT mask ---
#     matched_gt = {}
#     for pred_mask in pred_masks:
#         # mapped to a binary image where probability > pixel_threshold represents 1 and <=pixel_threshold represents 0
#         pred_binary = (pred_mask[0] > pixel_threshold).astype(np.uint8)
#         for i, gt_mask in enumerate(gt_masks):
#             # binarize GT mask using 0.5 threshold
#             gt_binary = (gt_mask > 0.5).astype(np.uint8)
#             intersection_mask = np.logical_and(pred_binary, gt_binary)
#             union_mask = np.logical_or(pred_binary, gt_binary)

#             # # Plot
#             # fig, axes = plt.subplots(1, 4, figsize=(16, 4))
#             # axes[0].imshow(pred_binary, cmap='Reds')
#             # axes[0].set_title('Prediction')
#             # axes[0].axis('off')

#             # axes[1].imshow(gt_binary, cmap='Greens')
#             # axes[1].set_title('Ground Truth')
#             # axes[1].axis('off')

#             # axes[2].imshow(intersection_mask, cmap='Blues')
#             # axes[2].set_title('Intersection')
#             # axes[2].axis('off')

#             # axes[3].imshow(union_mask, cmap='Purples')
#             # axes[3].set_title('Union')
#             # axes[3].axis('off')
#             # plt.tight_layout()
#             # plt.show()

#             # IOU formula
#             iou = intersection_mask.sum() / (union_mask.sum() + eps)
#             if iou > iou_threshold and matched_gt[i]:
#                 matched_gt[i]=iou
#             else:
#                 continue
#         # if best_gt_idx >= 0:
#         #     matched_gt.add(best_gt_idx)
#         total_iou += iou
#         count += 1

#     # --- Include unmatched GT as 0 IoU (false negatives) ---
#     for i in range(len(gt_masks)):
#         if i not in matched_gt:
#             total_iou += 0.0
#             count += 1

#     mean_iou = total_iou / count if count > 0 else 0.0
#     return total_iou, mean_iou, count


def evaluate_image(prediction, target, threshold=0.7, overlap_threshold=0.7):
    """
    Compute per-image TP, FP, FN, Precision, Recall based on IoU ≥ threshold
    """
    pred_boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    gt_boxes_tensor = target['boxes']
    gt_boxes = gt_boxes_tensor.cpu().numpy() if isinstance(gt_boxes_tensor, torch.Tensor) else gt_boxes_tensor

    matched_gt = set()
    tp, fp = 0, 0

    for box, score in zip(pred_boxes, scores):
        if score < threshold:
            continue
        matched = False
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            overlap = compute_overlap_area(box, gt)
            if (overlap / (compute_area(gt) + 1e-6)) >= overlap_threshold:
                matched = True
                matched_gt.add(i)
                tp += 1
                break
        if not matched:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return tp, fp, fn, precision, recall

def find_leaf_folders(root_dir):
    leaf_folders = []
    for current_dir, dirs, files in os.walk(root_dir):
        if not dirs:  
            normalized_path = current_dir.replace("\\", "/")
            leaf_folders.append(normalized_path)
    return leaf_folders

def train_test_val_split(sources, ratio=(0.8, 0.2, 0), shuffle=True):
    """
    Split the list of sources into train, test, and validation sets.

    Args:
        sources (list): List of folder paths.
        ratio (tuple): Split ratio for (train, test, val). Must sum to 1.0.
        shuffle (bool): Whether to shuffle before splitting.

    Returns:
        tuple: (train_list, test_list, val_list)
    """
    if not isinstance(ratio, (tuple, list)) or len(ratio) != 3:
        raise ValueError("ratio must be a tuple or list of 3 values (train, test, val).")
    if abs(sum(ratio) - 1.0) > 1e-6:
        raise ValueError("The sum of the split ratios must be 1.0")

    if shuffle:
        random.shuffle(sources)
    total = len(sources)
    train_end = int(ratio[0] * total)
    test_end = train_end + int(ratio[1] * total)

    train = sources[:train_end]
    test = sources[train_end:test_end]
    val = sources[test_end:]

    return train, test, val
def draw_binary_map(all_points_x, all_points_y, image_size=(5472, 3648)):
    # Ensure valid input
    if len(all_points_x) != len(all_points_y):
        raise ValueError("all_points_x and all_points_y must have the same length")

    # Create a blank black image (mode "L" for grayscale)
    mask = Image.new("L", image_size, 0)

    # Zip the points into (x, y) tuples
    polygon = list(zip(all_points_x, all_points_y))
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    # Convert to a NumPy binary array (0 and 1)
    mask_array = np.array(mask, dtype=np.uint8)

    return mask_array
def extract_json(filepath, image_name):
    # if filepath or image_name or image_jpg is None:
    #     raise ValueError(f"None object '{filepath, image_name, image_jpg}'")
    with open(filepath,"r") as f:
        data = json.load(f)
    item=None
    target_array=[]
    for d in data:
        if str(image_name) in str(d):
            item=d
        
    if item==None:
        raise ValueError(f"No value with image name '{image_name}' found in {filepath}")
    else:
        
        for an in data[item]["regions"]:
            # print(an)
            ann={}
            if an["shape_attributes"]["name"] == "rect":
                min_x=an["shape_attributes"]["x"]
                min_y=an["shape_attributes"]["y"]
                h=an["shape_attributes"]["height"]
                w=an["shape_attributes"]["width"]
                ann["bbox"]=[min_x, min_y, w, h]
                target_array.append(ann)
            else:
                all_points_x=an["shape_attributes"]["all_points_x"]
                all_points_y=an["shape_attributes"]["all_points_y"]
                min_x = min(all_points_x)
                max_x = max(all_points_x)
                min_y = min(all_points_y)
                max_y = max(all_points_y)
                bbox_array=[min_x, min_y, max_x-min_x, max_y-min_y]
                ann["bbox"]=bbox_array
                ann['category_id']=1
                target_array.append(ann)
    return target_array

def create_folder(path, deletion=True):
    if deletion==True:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
