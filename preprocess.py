# preprocess.py
import os
from PIL import Image

BASE_DIR = r"C:\Users\Lenovo Yoga\Desktop\WasteProject"  # EDIT to your folder
SRC_DIR = os.path.join(BASE_DIR, "datasets", "waste_4class")
DST_DIR = os.path.join(BASE_DIR, "datasets", "waste_4class_processed")
TARGET_SIZE = (224,224)

os.makedirs(DST_DIR, exist_ok=True)

def is_image(fn):
    return fn.lower().endswith(('.jpg','.jpeg','.png','.bmp'))

for cls in sorted(os.listdir(SRC_DIR)):
    scls = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(scls): continue
    dcls = os.path.join(DST_DIR, cls)
    os.makedirs(dcls, exist_ok=True)
    for fn in os.listdir(scls):
        if not is_image(fn): continue
        src = os.path.join(scls, fn)
        dst = os.path.join(dcls, fn)
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE, Image.BILINEAR)
                im.save(dst, "JPEG", quality=85)
        except Exception as e:
            print("skip", src, e)

print("Done. Processed images:", DST_DIR)
