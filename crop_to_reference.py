# Usage
# -----
#   python crop_to_reference.py --reference objects_shots/egg.png \
#       --images-dir objects_shots --out-dir objects_shots/cropped
"""
Center-crop a set of heterogeneous images to match a reference image's
pixel dimensions.

For every image in --images-dir, center-crops it to the reference's
(width, height). If an image is smaller than the reference in a given
dimension, that dimension is left untouched (cropping can only shrink an
image, never enlarge it) -- run this on already similarly-sized images if
you need every output to be pixel-identical in size.

Originals are never modified: cropped copies are written to --out-dir
under the same filenames.
"""

import argparse
import os

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    crop_w = min(w, target_w)
    crop_h = min(h, target_h)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to the reference image (defines target width/height).")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing the images to crop (non-recursive).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Defaults to <images-dir>/cropped.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.images_dir, "cropped")
    os.makedirs(out_dir, exist_ok=True)

    ref_img = Image.open(args.reference)
    target_w, target_h = ref_img.size
    ref_name = os.path.basename(args.reference)
    print(f"Reference: {args.reference} ({target_w}x{target_h})")

    for fname in sorted(os.listdir(args.images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        src_path = os.path.join(args.images_dir, fname)
        out_path = os.path.join(out_dir, fname)

        if fname == ref_name:
            ref_img.save(out_path)
            print(f"  {fname}: reference, copied unchanged -> {out_path}")
            continue

        img = Image.open(src_path)
        cropped = center_crop(img, target_w, target_h)
        cropped.save(out_path)
        print(f"  {fname}: {img.size} -> {cropped.size}  saved to {out_path}")


if __name__ == "__main__":
    main()
