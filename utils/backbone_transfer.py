import torch

from ultralytics import YOLO
from ultralytics.nn.modules.custom import OClipBottleneck, OClipResLayer, OClipStem

CKPT_PATH = "resnet50-oclip-7ba0c533.pth"


def strip_prefix(k: str) -> str:
    for p in (
        "visual.",
        "backbone.",
        "model.visual.",
        "module.visual.",
        "encoder.visual.",
        "trunk.",
        "img_encoder.",
    ):
        if k.startswith(p):
            return k[len(p):]
    return k


def load_oclip_state():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    raw_sd = ckpt.get("state_dict", ckpt)
    sd = {strip_prefix(k): v for k, v in raw_sd.items()}
    return sd


def copy_conv_scaled(tgt: torch.nn.Parameter, src: torch.Tensor):
    """
    Copy as much as possible from src -> tgt for Conv2d weights with possible
    channel mismatch due to width scaling.

    tgt: (out_t, in_t, kH, kW)
    src: (out_s, in_s, kH, kW)
    """
    out_t, in_t = tgt.shape[:2]
    out_s, in_s = src.shape[:2]
    out_c = min(out_t, out_s)
    in_c = min(in_t, in_s)
    with torch.no_grad():
        tgt.zero_()
        tgt[:out_c, :in_c, ...].copy_(src[:out_c, :in_c, ...])


def copy_bn_scaled(bn: torch.nn.BatchNorm2d, sd: dict, prefix: str):
    """
    Copy as much as possible from CLIP BN -> YOLO BN.
    bn: target BatchNorm2d
    sd: CLIP state_dict
    prefix: e.g. 'layer1.0.bn1'
    """
    for attr in ["weight", "bias", "running_mean", "running_var"]:
        key = f"{prefix}.{attr}"
        if key not in sd:
            continue
        src = sd[key]
        tgt = getattr(bn, attr)
        n = min(tgt.shape[0], src.shape[0])
        with torch.no_grad():
            tgt.zero_()
            tgt[:n].copy_(src[:n])


def init_backbone_weights(det_model, sd: dict):
    """
    det_model: YOLO Model object (yolo.model)
    sd: CLIP ResNet state_dict with keys like layer1.0.conv1.weight, ...
    """
    seq = det_model.model

    # Backbone stages according to your YAML:
    # 0: OClipStem
    # 1: OClipResLayer -> CLIP layer1
    # 2: OClipResLayer -> CLIP layer2
    # 3: OClipResLayer -> CLIP layer3
    # 4: OClipResLayer -> CLIP layer4
    stages = [seq[1], seq[2], seq[3], seq[4]]

    for stage_idx, layer in enumerate(stages, start=1):
        blocks = list(layer.layer)  # OClipBottleneck blocks
        for b_idx, block in enumerate(blocks):
            base = f"layer{stage_idx}.{b_idx}"

            # ----- main path conv1 / bn1 -----
            copy_conv_scaled(block.conv1.weight, sd[f"{base}.conv1.weight"])
            copy_bn_scaled(block.bn1, sd, f"{base}.bn1")

            # ----- conv2 / bn2 -----
            copy_conv_scaled(block.conv2.weight, sd[f"{base}.conv2.weight"])
            copy_bn_scaled(block.bn2, sd, f"{base}.bn2")

            # ----- conv3 / bn3 -----
            copy_conv_scaled(block.conv3.weight, sd[f"{base}.conv3.weight"])
            copy_bn_scaled(block.bn3, sd, f"{base}.bn3")

            # ----- downsample branch -----
            if block.downsample is not None:
                conv = block.downsample[1]
                bn = block.downsample[2]
                ds_base = f"{base}.downsample"

                # Two possible CLIP layouts:
                # 1) standard: downsample.0 = Conv, downsample.1 = BN
                # 2) avg_down: downsample.0 = AvgPool, downsample.1 = Conv, downsample.2 = BN
                if f"{ds_base}.0.weight" in sd:  # standard
                    conv_src = f"{ds_base}.0.weight"
                    bn_prefix = f"{ds_base}.1"
                else:  # avg_down
                    conv_src = f"{ds_base}.1.weight"
                    bn_prefix = f"{ds_base}.2"

                copy_conv_scaled(conv.weight, sd[conv_src])
                copy_bn_scaled(bn, sd, bn_prefix)


if __name__ == "__main__":
    sd = load_oclip_state()

    seg = YOLO("yolo11n-seg.yml")
    init_backbone_weights(seg.model, sd)
    seg.save("yolo11n-seg.pt")

    print("Saved initialized models.")
