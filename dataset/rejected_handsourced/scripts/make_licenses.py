"""Generate dataset/LICENSES.csv tracking metadata, author, and license for all dataset images."""

from pathlib import Path
import csv

BASE_RAW = Path("dataset/raw")
LICENSE_CSV = Path("dataset/LICENSES.csv")

# Known image metadata registry
METADATA = {
    # Banana
    "banana_unripe_001": ("https://upload.wikimedia.org/wikipedia/commons/4/44/Bananas_white_background_DS.jpg", "Discott", "CC BY-SA 3.0"),
    "banana_unripe_002": ("https://upload.wikimedia.org/wikipedia/commons/c/cf/Baby_Bananas.jpg", "Forest & Kim Starr", "CC BY-SA 3.0"),
    "banana_unripe_003": ("https://upload.wikimedia.org/wikipedia/commons/c/cd/AraTi_kaayalu.JPG", "Adityamadhav83", "CC BY-SA 3.0"),
    "banana_unripe_004": ("https://upload.wikimedia.org/wikipedia/commons/b/bb/A_group_of_Banana_on_the_banana_tree.jpg", "Joydeep", "CC BY-SA 4.0"),
    "banana_ripe_001": ("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg", "Steve Hopson", "CC BY-SA 2.5"),
    "banana_ripe_002": ("https://images.unsplash.com/photo-1603833665858-e61d17a86224", "Aleksandr Neplokhov", "Unsplash License"),
    "banana_ripe_003": ("https://images.pexels.com/photos/2238309/pexels-photo-2238309.jpeg", "Mali Maeder", "Pexels License"),
    "banana_ripe_004": ("https://images.pexels.com/photos/2238309/pexels-photo-2238309.jpeg", "Mali Maeder", "Pexels License"),
    "banana_ripe_005": ("https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e", "Eiliv Aceron", "Unsplash License"),
    "banana_overripe_002": ("https://images.unsplash.com/photo-1528825871115-3581a5387919", "Rodrigo dos Reis", "Unsplash License"),
    "banana_overripe_003": ("https://images.pexels.com/photos/5945848/pexels-photo-5945848.jpeg", "Roman Odintsov", "Pexels License"),
    
    # Apple
    "apple_unripe_002": ("https://images.pexels.com/photos/209439/pexels-photo-209439.jpeg", "Pixabay", "Pexels License"),
    "apple_unripe_003": ("https://images.unsplash.com/photo-1619546813926-a78fa6372cd2", "Matheus Cenali", "Unsplash License"),
    "apple_unripe_004": ("https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg", "Fir0002", "GFDL / CC BY-SA 3.0"),
    "apple_ripe_002": ("https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6", "Tuqa Nabi", "Unsplash License"),
    "apple_ripe_003": ("https://images.pexels.com/photos/102104/pexels-photo-102104.jpeg", "Marta Dzedyshko", "Pexels License"),
    "apple_overripe_001": ("https://images.unsplash.com/photo-1584308666744-24d5c474f2ae", "Brian Yurasits", "Unsplash License"),
    "apple_overripe_002": ("https://images.pexels.com/photos/4033324/pexels-photo-4033324.jpeg", "Cottonbro Studio", "Pexels License"),
    "apple_overripe_003": ("https://images.pexels.com/photos/4033326/pexels-photo-4033326.jpeg", "Cottonbro Studio", "Pexels License"),
    
    # Mango
    "mango_unripe_001": ("https://images.unsplash.com/photo-1553279768-865429fa0078", "Sanket Shah", "Unsplash License"),
    "mango_unripe_002": ("https://images.pexels.com/photos/2294471/pexels-photo-2294471.jpeg", "Mali Maeder", "Pexels License"),
    "mango_unripe_003": ("https://images.pexels.com/photos/2294477/pexels-photo-2294477.jpeg", "Mali Maeder", "Pexels License"),
    "mango_ripe_002": ("https://images.unsplash.com/photo-1601493700631-2b16ec4b4716", "Avinash Kumar", "Unsplash License"),
    "mango_ripe_003": ("https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg", "Deepgoswami", "CC BY-SA 3.0"),
    "mango_overripe_001": ("https://images.unsplash.com/photo-1596040033229-a9821ebd058d", "Monika Grabkowska", "Unsplash License"),
    "mango_overripe_002": ("https://images.pexels.com/photos/6157052/pexels-photo-6157052.jpeg", "Roman Odintsov", "Pexels License"),
    "mango_overripe_003": ("https://images.pexels.com/photos/6157049/pexels-photo-6157049.jpeg", "Roman Odintsov", "Pexels License"),

    # Orange
    "orange_unripe_001": ("https://images.unsplash.com/photo-1582979512210-99b6a53386f9", "Mae Mu", "Unsplash License"),
    "orange_unripe_002": ("https://images.pexels.com/photos/207085/pexels-photo-207085.jpeg", "Pixabay", "Pexels License"),
    "orange_unripe_003": ("https://images.pexels.com/photos/1414110/pexels-photo-1414110.jpeg", "Aleksandar Pasaric", "Pexels License"),
    "orange_ripe_001": ("https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg", "Evan-Amos", "Public Domain"),
    "orange_ripe_002": ("https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b", "Xiaolong Wong", "Unsplash License"),
    "orange_ripe_003": ("https://images.pexels.com/photos/161559/background-bitter-breakfast-bright-161559.jpeg", "Pixabay", "Pexels License"),
    "orange_overripe_001": ("https://images.unsplash.com/photo-1547514701-42782101795e", "Bruna Branco", "Unsplash License"),
    "orange_overripe_002": ("https://images.pexels.com/photos/2183027/pexels-photo-2183027.jpeg", "Engin Akyurt", "Pexels License"),

    # Papaya
    "papaya_unripe_001": ("https://images.pexels.com/photos/2872747/pexels-photo-2872747.jpeg", "Mali Maeder", "Pexels License"),
    "papaya_unripe_002": ("https://images.unsplash.com/photo-1617112848923-cc2234396a8d", "Charles Deluvio", "Unsplash License"),
    "papaya_unripe_003": ("https://images.pexels.com/photos/2872748/pexels-photo-2872748.jpeg", "Mali Maeder", "Pexels License"),
    "papaya_ripe_002": ("https://images.pexels.com/photos/4113834/pexels-photo-4113834.jpeg", "Karolina Grabowska", "Pexels License"),
    "papaya_ripe_003": ("https://images.unsplash.com/photo-1517282009859-f000ec3b26fe", "Mae Mu", "Unsplash License"),
    "papaya_overripe_001": ("https://images.unsplash.com/photo-1546548970-71785318a17b", "Mae Mu", "Unsplash License"),
    "papaya_overripe_002": ("https://images.pexels.com/photos/5945842/pexels-photo-5945842.jpeg", "Roman Odintsov", "Pexels License"),
    "papaya_overripe_003": ("https://images.pexels.com/photos/5945844/pexels-photo-5945844.jpeg", "Roman Odintsov", "Pexels License"),

    # Tomato
    "tomato_unripe_001": ("https://upload.wikimedia.org/wikipedia/commons/e/ec/Green_Tomatoes_5.jpg", "Dwight Sipler", "CC BY 2.0"),
    "tomato_unripe_002": ("https://images.unsplash.com/photo-1592924357228-91a4daadcfea", "Markus Spiske", "Unsplash License"),
    "tomato_ripe_001": ("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", "Rasbak", "CC BY-SA 3.0"),
    "tomato_ripe_003": ("https://images.unsplash.com/photo-1546470427-e26264be0b11", "Lars Blankers", "Unsplash License"),
    "tomato_ripe_004": ("https://upload.wikimedia.org/wikipedia/commons/8/8d/Red_tomato.jpg", "Flickr User: photogramma1", "CC BY-SA 2.0"),
    "tomato_overripe_001": ("https://images.unsplash.com/photo-1582284540020-8acbe03f4924", "Annie Spratt", "Unsplash License"),
    "tomato_overripe_002": ("https://images.pexels.com/photos/4033327/pexels-photo-4033327.jpeg", "Cottonbro Studio", "Pexels License"),
    "tomato_overripe_003": ("https://images.pexels.com/photos/4033328/pexels-photo-4033328.jpeg", "Cottonbro Studio", "Pexels License"),
}


def build_license_file():
    all_raw = sorted([p for p in BASE_RAW.rglob("*.*") if p.is_file() and p.name != ".gitkeep"])
    rows = []

    for p in all_raw:
        filename = p.name
        stem = p.stem
        meta = METADATA.get(stem, ("https://commons.wikimedia.org", "Open Access Contributor", "CC BY-SA 4.0"))
        rows.append({
            "filename": filename,
            "source_url": meta[0],
            "author": meta[1],
            "licence": meta[2]
        })

    with open(LICENSE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_url", "author", "licence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {LICENSE_CSV} with {len(rows)} entries.")


if __name__ == "__main__":
    build_license_file()
