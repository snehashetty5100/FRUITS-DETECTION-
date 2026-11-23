"""Fetch and Validate Additional Fruit Samples for Weak Classes

Target:
1. 3+ Genuine Green Unripe Bananas (banana/unripe)
2. 1+ Ripe Tomato (tomato/ripe)
3. 1+ Unripe Tomato (tomato/unripe)
4. Weak classes (apple/unripe, mango/ripe)

All images from Wikimedia Commons with open Creative Commons licenses:
- CC BY-SA 3.0 / CC BY-SA 4.0 / CC0 / Public Domain
"""

import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import resize_with_letterbox

BASE_RAW = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")

HEADERS = {
    "User-Agent": "FruitRipenessResearch/1.0 (College ML Project; mailto:student@university.edu)"
}

TARGET_DOWNLOADS = [
    # 1. Bananas (Unripe - Green)
    {
        "fruit": "banana",
        "stage": "unripe",
        "title": "File:Baby Bananas.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cf/Baby_Bananas.jpg",
        "license": "CC BY-SA 3.0",
        "source": "Wikimedia Commons"
    },
    {
        "fruit": "banana",
        "stage": "unripe",
        "title": "File:AraTi kaayalu.JPG",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cd/AraTi_kaayalu.JPG",
        "license": "CC BY-SA 3.0",
        "source": "Wikimedia Commons"
    },
    {
        "fruit": "banana",
        "stage": "unripe",
        "title": "File:A group of Banana on the banana tree.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/bb/A_group_of_Banana_on_the_banana_tree.jpg",
        "license": "CC BY-SA 4.0",
        "source": "Wikimedia Commons"
    },
    # 2. Tomato (Ripe)
    {
        "fruit": "tomato",
        "stage": "ripe",
        "title": "File:Red tomato.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8d/Red_tomato.jpg",
        "license": "CC BY-SA 3.0",
        "source": "Wikimedia Commons"
    },
    # 3. Tomato (Unripe - Green)
    {
        "fruit": "tomato",
        "stage": "unripe",
        "title": "File:Green tomato.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Green_Tomatoes_5.jpg",
        "license": "CC BY-SA 3.0",
        "source": "Wikimedia Commons"
    },
    # 4. Apple (Unripe - Green)
    {
        "fruit": "apple",
        "stage": "unripe",
        "title": "File:Granny smith and cross section.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg",
        "license": "CC BY-SA 3.0",
        "source": "Wikimedia Commons"
    }
]


def get_existing_hashes():
    hashes = set()
    for p in BASE_RAW.rglob("*.jpg"):
        try:
            with open(p, "rb") as f:
                hashes.add(hashlib.sha256(f.read()).hexdigest())
        except Exception:
            pass
    return hashes


def download_and_process():
    existing_hashes = get_existing_hashes()
    downloaded = []

    print("=== Fetching Additional Verified Fruit Samples ===")
    for item in TARGET_DOWNLOADS:
        fruit = item["fruit"]
        stage = item["stage"]
        target_raw = BASE_RAW / fruit / stage
        target_proc = PROCESSED_DIR / fruit / stage
        target_raw.mkdir(parents=True, exist_ok=True)
        target_proc.mkdir(parents=True, exist_ok=True)

        existing = sorted(target_raw.glob(f"{fruit}_{stage}_*.jpg"))
        next_idx = len(existing) + 1
        raw_filename = f"{fruit}_{stage}_{next_idx:03d}.jpg"
        proc_filename = f"{fruit}_{stage}_{next_idx:03d}.png"
        raw_path = target_raw / raw_filename
        proc_path = target_proc / proc_filename

        print(f"Downloading [{fruit}/{stage}] from {item['source']}: {item['title']}...")
        try:
            req = urllib.request.Request(item["url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = resp.read()

            file_hash = hashlib.sha256(data).hexdigest()
            if file_hash in existing_hashes:
                print(f"  [!] Skipped: Image hash already exists in raw dataset.")
                continue

            with open(raw_path, "wb") as f:
                f.write(data)

            # Validate integrity
            with Image.open(raw_path) as img:
                img.verify()

            # Letterbox process to 224x224
            with Image.open(raw_path) as img:
                rgb_arr = img.convert("RGB")
                import numpy as np
                padded, _, _ = resize_with_letterbox(np.array(rgb_arr), (224, 224))
                Image.fromarray(padded).save(proc_path, "PNG")

            existing_hashes.add(file_hash)
            downloaded.append({
                "filename": raw_filename,
                "fruit": fruit,
                "stage": stage,
                "raw_path": str(raw_path),
                "proc_path": str(proc_path),
                "license": item["license"],
                "source": item["source"],
                "title": item["title"]
            })
            print(f"  [✓] Saved & Letterboxed: {raw_filename} -> {proc_filename} (License: {item['license']})")

        except Exception as e:
            if raw_path.exists():
                raw_path.unlink()
            if proc_path.exists():
                proc_path.unlink()
            print(f"  [X] Failed {item['url']}: {e}")

    # Also letterbox process banana_unripe_001 if not yet processed
    b1_raw = BASE_RAW / "banana" / "unripe" / "banana_unripe_001.jpg"
    b1_proc = PROCESSED_DIR / "banana" / "unripe" / "banana_unripe_001.png"
    if b1_raw.exists() and not b1_proc.exists():
        import numpy as np
        with Image.open(b1_raw) as img:
            padded, _, _ = resize_with_letterbox(np.array(img.convert("RGB")), (224, 224))
            Image.fromarray(padded).save(b1_proc, "PNG")
        print("  [✓] Letterboxed banana_unripe_001.png")

    print(f"\nSuccessfully downloaded & processed {len(downloaded)} new verified samples.")
    return downloaded


if __name__ == "__main__":
    download_and_process()
