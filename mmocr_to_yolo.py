# import os
# import json
# import shutil
# from pathlib import Path
# import numpy as np
# import cv2


# def _order_tl_ccw(pts4: np.ndarray) -> np.ndarray:
#     """Order 4x2 corners: top-left first, then counter-clockwise."""
#     pts = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
#     # top (smallest y); among top two, tl has smaller x
#     idx = np.argsort(pts[:, 1])
#     top2, bot2 = pts[idx[:2]], pts[idx[2:]]
#     tl = top2[np.argmin(top2[:, 0])]
#     tr = top2[np.argmax(top2[:, 0])]
#     bl = bot2[np.argmin(bot2[:, 0])]
#     br = bot2[np.argmax(bot2[:, 0])]
#     quad = np.stack([tl, tr, br, bl], axis=0)
#     # ensure CCW
#     if np.cross(quad[1] - quad[0], quad[2] - quad[0]) < 0:
#         quad = quad[[0, 3, 2, 1]]
#     return quad


# def _poly_to_minarea_quad(poly_xy):
#     """poly_xy: [x1,y1,...] -> 4x2 min-area rectangle corners (float32)."""
#     pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
#     if pts.shape[0] < 4:
#         return None
#     if pts.shape[0] == 4:
#         return _order_tl_ccw(pts)
#     rect = cv2.minAreaRect(pts)       # ((cx,cy),(w,h),angle)
#     quad = cv2.boxPoints(rect)        # 4x2
#     return _order_tl_ccw(quad)


# def mmocr_to_yolo_obb(mmocr_json_path, dst_images_dir, dst_labels_dir, copy_images=True, class_id=0):
#     """
#     Convert MMOCR TextDet JSON (pixel polygons) -> YOLO-OBB labels (normalized quads).
#     Writes one .txt per image in dst_labels_dir. Optionally copies images to dst_images_dir.
#     """
#     dst_images_dir = Path(dst_images_dir)
#     dst_labels_dir = Path(dst_labels_dir)
#     dst_labels_dir.mkdir(parents=True, exist_ok=True)
#     if copy_images:
#         dst_images_dir.mkdir(parents=True, exist_ok=True)

#     root = Path(mmocr_json_path).parent
#     data = json.load(open(mmocr_json_path, 'r', encoding='utf-8'))

#     img_cnt, inst_cnt = 0, 0
#     for item in data['data_list']:
#         rel_img = item['img_path']
#         H, W = int(item.get('height', 0)), int(item.get('width', 0))

#         src_img = root / rel_img
#         if (H == 0 or W == 0) or (not src_img.exists()):
#             # fallback: read actual image to get size if fields missing
#             img = cv2.imread(str(src_img))
#             if img is None:
#                 print(f"[WARN] skip, image missing: {src_img}")
#                 continue
#             H, W = img.shape[:2]

#         stem = Path(rel_img).stem
#         out_txt = dst_labels_dir / f"{stem}.txt"
#         lines = []

#         for inst in item.get('instances', []):
#             if inst.get('ignore', False):
#                 continue
#             poly = inst.get('polygon')
#             if not poly or len(poly) < 8:
#                 continue

#             quad = _poly_to_minarea_quad(poly)
#             if quad is None:
#                 continue
#             # clip to image bounds
#             quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
#             quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
#             # normalize
#             qn = quad.astype(np.float32)
#             qn[:, 0] /= float(W)
#             qn[:, 1] /= float(H)
#             vals = qn.reshape(-1).tolist()
#             lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in vals))
#             inst_cnt += 1

#         # write label (empty file is okay)
#         with open(out_txt, 'w', encoding='utf-8') as f:
#             f.write("\n".join(lines))

#         if copy_images and src_img.exists():
#             # flatten; or preserve subdirs if you prefer
#             dst = dst_images_dir / Path(rel_img).name
#             if str(src_img) != str(dst):
#                 dst.parent.mkdir(parents=True, exist_ok=True)
#                 shutil.copy2(src_img, dst)

#         img_cnt += 1

#     # optional: save class list
#     with open(dst_labels_dir / 'classes.txt', 'w', encoding='utf-8') as f:
#         f.write("text\n")

#     print(f"Converted {img_cnt} images, {inst_cnt} OBBs → {dst_labels_dir}")


# mmocr_to_yolo_obb(
#     mmocr_json_path='data/textdet_test.json',
#     dst_images_dir='dataset/images/test',
#     dst_labels_dir='dataset/labels/test',
#     copy_images=True,  # set False if your images are already in place
#     class_id=0
# )


# mmocr_to_yolo_obb(
#     mmocr_json_path='data/textdet_train.json',
#     dst_images_dir='dataset/images/train',
#     dst_labels_dir='dataset/labels/train',
#     copy_images=True,  # set False if your images are already in place
#     class_id=0
# )

def visualize_yolo_obb(img_path, txt_path):
    import cv2
    import numpy as np
    im = cv2.imread(img_path)
    H, W = im.shape[:2]
    for line in open(txt_path, 'r'):
        p = line.strip().split()
        if len(p) != 9:
            continue
        xy = np.array(list(map(float, p[1:])), dtype=np.float32).reshape(4, 2)
        xy[:, 0] *= W
        xy[:, 1] *= H
        cv2.polylines(im, [xy.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imshow('yolo-obb', im)
    cv2.waitKey(0)


visualize_yolo_obb('dataset/images/test/crop_81_flatten_00834_jpg.rf.df715a0d9a5c9aa470080b9448ebdebe.jpg',
                   'dataset/labels/test/crop_81_flatten_00834_jpg.rf.df715a0d9a5c9aa470080b9448ebdebe.txt')
