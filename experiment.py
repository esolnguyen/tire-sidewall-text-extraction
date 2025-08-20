import torch
from ultralytics import YOLO

# yolo = YOLO('yolo_11.yml')
# pretrained_backbone = torch.load(
#     'yolov11_resnet50_init.pth',
#     map_location='cpu'
# )
# yolo.model.load_state_dict(pretrained_backbone)


# yolo.eval()
# with torch.no_grad():
#     dummy_input = torch.randn(1, 3, 640, 640)
#     output = yolo(dummy_input)
#     print(output)

# results = yolo.train(data='coco.yaml', epochs=100, imgsz=640)

# Load a pretrained YOLO model from 'best.pt'
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('best.pt')

# If you're on Apple Silicon, try device='mps' for speed; else 0 (CUDA) or 'cpu'
res = model.predict('data/tire/image_00001.jpg', conf=0.25, device='mps')[0]

# 1) Access rotated boxes
poly = res.obb.xyxyxyxy.cpu().numpy()      # (N, 4, 2) corners
xywhr = res.obb.xywhr.cpu().numpy()        # (N, 5): cx, cy, w, h, rot(rad)
cls = res.obb.cls.cpu().numpy().astype(int)
conf = res.obb.conf.cpu().numpy()

print(f"{len(conf)} detections")
for i in range(len(conf)):
    name = res.names[cls[i]]
    print(f"{name} {conf[i]:.2f} center=({xywhr[i,0]:.1f},{xywhr[i,1]:.1f}) "
          f"w={xywhr[i,2]:.1f} h={xywhr[i,3]:.1f} rot={xywhr[i,4]:.3f} rad")

# 2) Visualize & save annotated image
vis = res.plot()                 # draws rotated boxes
cv2.imwrite('pred.jpg', vis)
print("Saved viz to pred.jpg")

# 3) (Optional) Save YOLO-OBB label file for this image
H, W = res.orig_shape
stem = '1010'  # image stem
with open(f'{stem}.txt', 'w') as f:
    for i in range(len(conf)):
        q = poly[i].copy()
        q[:, 0] /= W
        q[:, 1] /= H          # normalize to [0,1]
        f.write(f"{cls[i]} " +
                " ".join(f"{v:.6f}" for v in q.reshape(-1)) + "\n")
