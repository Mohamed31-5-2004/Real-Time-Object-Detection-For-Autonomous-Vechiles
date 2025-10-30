# scripts/explore_coco.py
import os, json, random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from collections import Counter
import pandas as pd

# ---- USER CONFIG ----
BASE_DIR = r"E:\all e drive\AIU COMPUTER SCIENCE BACLERIOS\AIU COMPUTER SCIENCE SIXTH SEMESTER SPRING (24-25)\DEPI\Final Project\CocoDataSet\coco_preprocess"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANN_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_ANN = os.path.join(ANN_DIR, "instances_train2017.json")
VAL_ANN = os.path.join(ANN_DIR, "instances_val2017.json")
IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train2017")
IMG_DIR_VAL = os.path.join(DATA_DIR, "val2017")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "exploration")

SAMPLE_COUNT = 8

# ✅ Selected 24 classes for autonomous-driving dataset
SELECTED_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "bench", "parking meter",
    "umbrella", "backpack", "handbag", "tie", "cell phone", "dog", "cat",
    "horse", "bird", "skateboard", "boat", "suitcase"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
def explore(ann_file, img_dir, split_name="train"):
    print(f"\n=== Exploring {split_name.upper()} split ===")
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c['id']: c['name'] for c in cats}
    selected_cat_ids = coco.getCatIds(catNms=SELECTED_CLASSES)

    # --- Basic statistics ---
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    widths = [img['width'] for img in images]
    heights = [img['height'] for img in images]
    class_counts = Counter()
    for a in anns:
        if a['category_id'] in selected_cat_ids:
            class_counts[cat_id_to_name[a['category_id']]] += 1

    objs_per_image = [len(coco.getAnnIds(imgIds=img['id'], catIds=selected_cat_ids)) for img in images]

    summary = {
        "split": split_name,
        "num_images": len(images),
        "num_annotations_total": len(anns),
        "num_annotations_selected": sum(class_counts.values()),
        "image_width_mean": np.mean(widths),
        "image_height_mean": np.mean(heights),
        "avg_objects_selected_per_image": np.mean(objs_per_image)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, f"summary_{split_name}.csv"), index=False)
    print("Summary:", summary)

    # --- Visualization section ---
    sns.set_style("whitegrid")

    # 1. Class distribution
    plt.figure(figsize=(10,6))
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    sns.barplot(x=counts, y=classes, palette="crest")
    plt.title(f"Class Distribution - {split_name}")
    plt.xlabel("Object count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{split_name}.png"))
    plt.close()

    # 2. Objects per image histogram
    plt.figure(figsize=(7,5))
    plt.hist(objs_per_image, bins=30, color="royalblue")
    plt.xlabel("Objects per image")
    plt.ylabel("Number of images")
    plt.title(f"Objects per Image Histogram ({split_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"objects_per_image_{split_name}.png"))
    plt.close()

    # 3. Image dimension distribution
    plt.figure(figsize=(6,6))
    plt.scatter(widths, heights, s=10, alpha=0.4)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(f"Image Resolution Scatter ({split_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"image_resolution_{split_name}.png"))
    plt.close()

    # --- Sample image visualization ---
    print("Generating example annotated images...")
    sample_img_ids = random.sample(img_ids, min(SAMPLE_COUNT, len(img_ids)))
    for i, img_id in enumerate(sample_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        anns_ids = coco.getAnnIds(imgIds=img_id, catIds=selected_cat_ids)
        anns_sel = coco.loadAnns(anns_ids)
        for ann in anns_sel:
            x, y, w, h = map(int, ann['bbox'])
            cat_name = cat_id_to_name[ann['category_id']]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, cat_name, (x, max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = os.path.join(OUTPUT_DIR, f"sample_{split_name}_{i}.jpg")
        cv2.imwrite(out_path, img)

    # --- Automatic insights ---
    print("\n📊 Key Insights:")
    top5 = class_counts.most_common(5)
    print(f"Top 5 most common objects: {top5}")
    rare5 = class_counts.most_common()[-5:]
    print(f"Least common objects: {rare5}")
    print(f"Average objects per image: {summary['avg_objects_selected_per_image']:.2f}")
    print(f"Average image size: {summary['image_width_mean']:.0f}×{summary['image_height_mean']:.0f}")

    print("\n💡 Analysis Q&A:")
    print("Q1: Which objects dominate the dataset?")
    print(f"A1: {top5[0][0]} appears most frequently, suggesting focus on urban scenes.")
    print("Q2: Are there underrepresented classes?")
    print(f"A2: The rarest are {', '.join([c for c,_ in rare5])}, consider augmenting them.")
    print("Q3: How dense are scenes?")
    print(f"A3: On average {summary['avg_objects_selected_per_image']:.1f} relevant objects per image.")
    print("Q4: Are image sizes consistent?")
    print(f"A4: Widths range {min(widths)}–{max(widths)}, heights {min(heights)}–{max(heights)}.")
    print("Q5: What do examples show?")
    print("A5: Sample images saved with bounding boxes illustrate diversity in object placement and scale.\n")

    return summary, class_counts


if __name__ == "__main__":
    print("Starting Dataset Exploration ...")
    explore(TRAIN_ANN, IMG_DIR_TRAIN, "train")
    explore(VAL_ANN, IMG_DIR_VAL, "val")
    print(f"\n✅ Exploration finished. Check visual reports in:\n{OUTPUT_DIR}")

