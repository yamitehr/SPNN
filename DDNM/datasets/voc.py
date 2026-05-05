"""Pascal VOC validation dataloader for DDNM detection runs.

Returns images in [0, 1] RGB, shape [3, H, W]. This matches the other DDNM
dataset loaders (CelebA, ImageNet) which use torchvision's ToTensor() and
also output [0, 1]; the diffusion pipeline's `data_transform(rescaled=True)`
then maps to [-1, 1] for the diffusion model.

The detection-domain conversion (RGB→BGR, ImageNet normalization) happens
inside the A() function in diffusion.py, not here.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


class VOCValForDDNM(data.Dataset):
    """Reads VOC2007 test images, returns [0, 1] RGB tensors square center-
    cropped to image_size × image_size. Matches the [0,1] convention used by
    the rest of DDNM's dataset loaders; data_transform() in the pipeline
    handles the [0,1]→[-1,1] rescaling for the diffusion model.
    """

    def __init__(self, data_dir, image_size=256):
        self.image_size = image_size
        self.img_dir = os.path.join(data_dir, "voc", "images")
        annot_path = os.path.join(
            data_dir, "voc", "annotations", "pascal_test2007.json"
        )
        with open(annot_path) as f:
            ann = json.load(f)
        self.images = ann["images"]  # list of {file_name, id, height, width, ...}

        # Index annotations by image_id and category_id → name, so the DDNM
        # grid-rendering code can call get_gt_in_image_coords(image_id) to
        # overlay GT boxes on the (cropped+resized) original image.
        self._image_by_id = {im["id"]: im for im in self.images}
        self._anns_by_image_id = {}
        for a in ann.get("annotations", []):
            self._anns_by_image_id.setdefault(a["image_id"], []).append(a)
        self._cat_name = {c["id"]: c["name"] for c in ann.get("categories", [])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, info["file_name"]))
        if img is None:
            raise FileNotFoundError(
                f"could not read {os.path.join(self.img_dir, info['file_name'])}"
            )
        h, w = img.shape[:2]
        # Center-crop to square, then resize to image_size×image_size
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_AREA)
        # cv2 returns BGR; the diffusion domain is RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Return [0, 1] (NOT [-1, 1]) — data_transform() in the diffusion
        # pipeline handles the rescaling, matching CelebA/ImageNet datasets.
        img = img.astype(np.float32) / 255.0  # [0, 255] → [0, 1]
        img = img.transpose(2, 0, 1)  # HWC → CHW
        # `class` is unused for detection runs; return image_id so callers can
        # tie a result back to the original VOC image if they want.
        return torch.from_numpy(img.copy()), int(info["id"])

    def get_gt_in_image_coords(self, image_id):
        """Returns GT boxes for `image_id` in the SAME coordinate frame as
        __getitem__'s output (i.e. after the same center-crop + resize to
        image_size × image_size that __getitem__ applies).

        Output: list of (class_name, (x1, y1, x2, y2)) tuples.

        Used by the DDNM grid renderer to overlay GT boxes on the original
        image alongside the detector's predictions.
        """
        info = self._image_by_id[image_id]
        h, w = info["height"], info["width"]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        scale = self.image_size / s

        out = []
        for a in self._anns_by_image_id.get(image_id, []):
            x, y, bw, bh = a["bbox"]  # COCO format: [x, y, w, h] in image coords
            # Translate from image frame to crop frame
            x1, y1 = x - x0, y - y0
            x2, y2 = x + bw - x0, y + bh - y0
            # Clip to crop bounds [0, s] × [0, s]
            x1 = max(0.0, min(float(s), x1))
            y1 = max(0.0, min(float(s), y1))
            x2 = max(0.0, min(float(s), x2))
            y2 = max(0.0, min(float(s), y2))
            # Drop boxes that fall entirely outside the central square crop
            if x2 <= x1 or y2 <= y1:
                continue
            # Scale crop frame → image_size frame
            x1 *= scale; y1 *= scale; x2 *= scale; y2 *= scale
            name = self._cat_name.get(a["category_id"], str(a["category_id"]))
            out.append((name, (x1, y1, x2, y2)))
        return out
