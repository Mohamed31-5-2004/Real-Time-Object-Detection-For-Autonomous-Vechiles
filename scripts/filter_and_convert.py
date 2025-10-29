# scripts/filter_and_convert.py
import os, json, shutil
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
import numpy as np

# ---- USER CONFIG (Final Stable Version) ----

import os

# ✅ Absolute path to your coco_preprocess folder
BASE_DIR = r"E:\all e drive\AIU COMPUTER SCIENCE BACLERIOS\AIU COMPUTER SCIENCE SIXTH SEMESTER SPRING (24-25)\DEPI\Final Project\CocoDataSet\coco_preprocess"

DATA_DIR = os.path.join(BASE_DIR, "data")
ANN_DIR = os.path.join(DATA_DIR, "annotations")

TRAIN_ANN = os.path.join(ANN_DIR, "instances_train2017.json")
VAL_ANN = os.path.join(ANN_DIR, "instances_val2017.json")

IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train2017")
IMG_DIR_VAL = os.path.join(DATA_DIR, "val2017")

OUT = os.path.join(BASE_DIR, "outputs", "filtered_dataset")
IMG_SIZE = 640  # final size (square)

# ✅ Debug check — confirms paths exist
print("TRAIN_ANN exists:", os.path.exists(TRAIN_ANN))
print("VAL_ANN exists:", os.path.exists(VAL_ANN))
print("IMG_DIR_TRAIN exists:", os.path.exists(IMG_DIR_TRAIN))
print("IMG_DIR_VAL exists:", os.path.exists(IMG_DIR_VAL))


# -----------------------------------------------
# Selected COCO Classes for Autonomous Driving
# -----------------------------------------------
SELECTED_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "bench", "parking meter",
    "umbrella", "backpack", "handbag", "tie", "cell phone", "dog", "cat",
    "horse", "bird", "skateboard", "boat", "suitcase"
]


    
SPLIT_MAP = {
    "train": (TRAIN_ANN, IMG_DIR_TRAIN),
    "val": (VAL_ANN, IMG_DIR_VAL)
}
# ----------------------

os.makedirs(OUT, exist_ok=True)
for s in ["train", "val"]:
    os.makedirs(os.path.join(OUT, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, s, "labels"), exist_ok=True)

# Build mapping COCO_cat_id -> new compact id (0..n-1)
def build_cat_mapping(coco, selected_names):
    # get category IDs directly from names
    selected_cat_ids = coco.getCatIds(catNms=selected_names)
    print("Selected cat IDs:", selected_cat_ids)
    if not selected_cat_ids:
        print("⚠️ Warning: No matching categories found! Check class names.")
    mapping = {cid: i for i, cid in enumerate(selected_cat_ids)}
    return mapping, selected_cat_ids




def process_split(split):
    ann_file, img_dir = SPLIT_MAP[split]
    print(f"\n=== Processing {split} ===")
    print("Annotation file used:", ann_file)
    print("Exists?", os.path.exists(ann_file))
    coco = COCO(ann_file)
    mapping, selected_cat_ids = build_cat_mapping(coco, SELECTED_CLASSES)
    selected_cat_names = [coco.loadCats([cid])[0]['name'] for cid in selected_cat_ids]

    test_car_id = coco.getCatIds(catNms=["car"])
    test_car_imgs = coco.getImgIds(catIds=test_car_id)
    print("Debug - Car ID:", test_car_id, "| Images found:", len(test_car_imgs))

    # Collect images containing ANY of the selected categories
    img_ids = list(set().union(*[coco.getImgIds(catIds=[cid]) for cid in selected_cat_ids]))
    print(f"{split}: {len(img_ids)} images with selected classes")

    for img_id in tqdm(img_ids):
        img_meta = coco.loadImgs(img_id)[0]
        fname = img_meta['file_name']
        src_path = os.path.join(img_dir, fname)
        if not os.path.exists(src_path):
            continue
        # open and resize
        im = Image.open(src_path).convert('RGB')
        orig_w, orig_h = im.size
        im_resized = im.resize((IMG_SIZE, IMG_SIZE))
        out_img_path = os.path.join(OUT, split, "images", fname)
        im_resized.save(out_img_path, quality=95)

        # gather anns only for selected classes
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=selected_cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        yolo_lines = []
        for ann in anns:
            coco_cat_id = ann['category_id']
            if coco_cat_id not in mapping:
                continue
            new_class = mapping[coco_cat_id]
            x, y, w, h = ann['bbox']  # COCO: top-left x,y,width,height
            # normalize to resized image:
            x_c = x + w/2.0
            y_c = y + h/2.0
            x_c /= orig_w
            y_c /= orig_h
            w /= orig_w
            h /= orig_h
            # convert to resized coordinates normalized (still 0..1 so OK)
            # YOLO expects: class x_center y_center width height (all normalized)
            yolo_lines.append(f"{new_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # write labels if any
        label_path = os.path.join(OUT, split, "labels", os.path.splitext(fname)[0] + ".txt")
        with open(label_path, "w") as f:
            if len(yolo_lines) > 0:
                f.write("\n".join(yolo_lines))

    # build names list in order of mapping (we need names list by new_id)
    inv_map = {v:k for k,v in mapping.items()}
    names = []
    for new_id in range(len(inv_map)):
        coco_id = inv_map[new_id]
        names.append(coco.loadCats([coco_id])[0]['name'])
    return names





if __name__ == "__main__":
    # process both splits
    names = None
    for s in ["val"]:
        names = process_split(s)

    # write data.yaml
    data_yaml = {
        'train': os.path.abspath(os.path.join(OUT, "train", "images")),
        'val': os.path.abspath(os.path.join(OUT, "val", "images")),
        'test': os.path.abspath(os.path.join(OUT, "val", "images")),  # use val for test if no separate test
        'nc': len(names),
        'names': names
    }
    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    print("Done. data.yaml created at", yaml_path)
    print("Classes:", names)
