"""Replace mismatched images in test fixture with verified, license-free Wikimedia Commons images."""

import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

HEADERS = {"User-Agent": "FruitRipenessProject/1.0 (academic research; snehashetty510@gmail.com)"}

VERIFIED_SOURCES = {
    ("tomato", "unripe", "tomato_unripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/e/ec/Green_Tomatoes_01.jpg",
    ("tomato", "unripe", "tomato_unripe_002.jpg"): "https://upload.wikimedia.org/wikipedia/commons/d/d4/Unripe_tomatoes.jpg",
    ("tomato", "ripe", "tomato_ripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg",
    ("tomato", "ripe", "tomato_ripe_002.jpg"): "https://upload.wikimedia.org/wikipedia/commons/a/a2/Tomato_je.jpg",
    ("banana", "unripe", "banana_unripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/f/f6/Green_raw_bananas.jpg",
    ("banana", "unripe", "banana_unripe_002.jpg"): "https://upload.wikimedia.org/wikipedia/commons/b/b3/Green_Bananas.jpg",
    ("banana", "ripe", "banana_ripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
    ("apple", "unripe", "apple_unripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_Smith_apple.jpg",
    ("orange", "ripe", "orange_ripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/7/7b/Orange-Whole-%26-Split.jpg",
    ("papaya", "unripe", "papaya_unripe_001.jpg"): "https://upload.wikimedia.org/wikipedia/commons/3/36/Green_Papaya.jpg",
}

def download_and_verify():
    for (fruit, stage, filename), url in VERIFIED_SOURCES.items():
        dest = Path("dataset/raw") / fruit / stage / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                if len(data) > 5000:
                    with open(dest, "wb") as f:
                        f.write(data)
                    # Verify image can be opened
                    with Image.open(dest) as img:
                        w, h = img.size
                        arr = np.array(img.convert("RGB"))
                        r, g, b = arr.mean(axis=(0, 1))
                        print(f"Downloaded and verified: {fruit}/{stage}/{filename} ({w}x{h}) - Mean RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    download_and_verify()
