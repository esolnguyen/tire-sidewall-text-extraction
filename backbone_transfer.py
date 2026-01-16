from ultralytics import YOLO
import torch.nn as nn
import torch

CKPT_PATH = "models/resnet50-oclip.pth"


def strip_prefix(k: str) -> str:
    # Các prefix thường gặp trong các bộ weight pre-trained
    prefixes = [
        "visual.", "backbone.", "model.visual.", "module.visual.",
        "encoder.visual.", "trunk.", "img_encoder.", "module."
    ]
    for p in prefixes:
        if k.startswith(p):
            return k[len(p):]
    return k


def load_oclip_state():
    print(f"Loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # Xử lý trường hợp weight nằm trong key 'state_dict' hoặc 'model'
    raw_sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    sd = {strip_prefix(k): v for k, v in raw_sd.items()}
    return sd


def copy_conv_scaled(tgt: torch.nn.Parameter, src: torch.Tensor):
    """
    Copy weight từ src -> tgt, xử lý việc lệch số kênh (width scaling)
    và lệch kernel size (nếu có, ví dụ 7x7 -> 3x3).
    """
    out_t, in_t, kh_t, kw_t = tgt.shape
    out_s, in_s, kh_s, kw_s = src.shape

    # 1. Xử lý số kênh (Channel Slicing)
    out_c = min(out_t, out_s)
    in_c = min(in_t, in_s)

    # 2. Xử lý Kernel Size (Spatial Slicing/Centering)
    # Nếu kernel src to hơn tgt (vd 7x7 -> 3x3), lấy phần giữa
    start_h = (kh_s - kh_t) // 2
    start_w = (kw_s - kw_t) // 2

    with torch.no_grad():
        # Chỉ zero phần kênh sẽ copy để an toàn, phần thừa giữ nguyên init
        tgt[:out_c, :in_c, :, :].zero_()

        # Copy vùng trung tâm
        tgt[:out_c, :in_c, :, :].copy_(
            src[:out_c, :in_c, start_h:start_h+kh_t, start_w:start_w+kw_t]
        )


def copy_bn_scaled(bn: torch.nn.BatchNorm2d, sd: dict, prefix: str):
    for attr in ["weight", "bias", "running_mean", "running_var"]:
        key = f"{prefix}.{attr}"
        if key not in sd:
            continue
        src = sd[key]
        tgt = getattr(bn, attr)
        n = min(tgt.shape[0], src.shape[0])
        with torch.no_grad():
            tgt[:n].copy_(src[:n])


def init_stem_weights(stem_layer, sd: dict):
    mappings = [
        (stem_layer.conv1, stem_layer.bn1, [
         "stem.0", "conv1", "backbone.stem.0"]),  # Layer 1
        (stem_layer.conv2, stem_layer.bn2, [
         "stem.2", "stem.1", "backbone.stem.2"]),  # Layer 2 (nếu có)
        (stem_layer.conv3, stem_layer.bn3, [
         "stem.4", "stem.2", "backbone.stem.4"]),  # Layer 3 (nếu có)
    ]

    for conv_target, bn_target, possible_names in mappings:
        found = False
        for name in possible_names:
            conv_key = f"{name}.weight"
            if conv_key in sd:
                print(
                    f"  -> Found Stem map: {name} -> Conv ({conv_target.kernel_size})")
                copy_conv_scaled(conv_target.weight, sd[conv_key])

                # Tìm BN tương ứng (thường là cùng tên nhưng là .1 hoặc bn1)
                # Hack: đoán tên BN dựa trên tên Conv
                bn_prefix = name
                if "conv1" in name:
                    bn_prefix = "bn1"  # ResNet chuẩn
                elif "stem.0" in name:
                    bn_prefix = "stem.1"  # ResNet-vd structure
                elif "stem.2" in name:
                    bn_prefix = "stem.3"

                copy_bn_scaled(bn_target, sd, bn_prefix)
                found = True
                break

        if not found:
            print(
                f"  ! Warning: Could not find weight for a Stem layer. It will use random init.")


def init_backbone_weights(det_model, sd: dict):
    seq = det_model.model

    # 1. Transfer Stem
    if hasattr(seq[0], 'conv1'):
        init_stem_weights(seq[0], sd)
    else:
        print("Warning: seq[0] does not look like a Stem.")

    # 2. Transfer các Stage (Bottlenecks)
    stage_mapping = [
        (1, "layer1"),
        (2, "layer2"),
        (4, "layer3"),
        (6, "layer4"),
    ]

    for model_idx, clip_layer_name in stage_mapping:
        layer = seq[model_idx]
        if not hasattr(layer, 'layer'):
            continue

        blocks = list(layer.layer)
        print(
            f"Processing Stage {model_idx} -> CLIP {clip_layer_name} ({len(blocks)} blocks)")

        for b_idx, block in enumerate(blocks):
            base = f"{clip_layer_name}.{b_idx}"

            # Kiểm tra xem block này có tồn tại trong Source không
            if f"{base}.conv1.weight" not in sd:
                print(
                    f"  Stop at block {b_idx}: Source checkpoint run out of blocks here.")
                break

            # Copy Conv1/BN1
            copy_conv_scaled(block.conv1.weight, sd[f"{base}.conv1.weight"])
            copy_bn_scaled(block.bn1, sd, f"{base}.bn1")

            # Copy Conv2/BN2
            copy_conv_scaled(block.conv2.weight, sd[f"{base}.conv2.weight"])
            copy_bn_scaled(block.bn2, sd, f"{base}.bn2")

            # Copy Conv3/BN3
            copy_conv_scaled(block.conv3.weight, sd[f"{base}.conv3.weight"])
            copy_bn_scaled(block.bn3, sd, f"{base}.bn3")

            # Downsample
            if block.downsample is not None:
                ds_base = f"{base}.downsample"
                conv_ds, bn_ds = None, None
                for m in block.downsample:
                    if isinstance(m, nn.Conv2d):
                        conv_ds = m
                    if isinstance(m, nn.BatchNorm2d):
                        bn_ds = m

                if conv_ds:
                    if f"{ds_base}.0.weight" in sd:
                        copy_conv_scaled(
                            conv_ds.weight, sd[f"{ds_base}.0.weight"])
                        copy_bn_scaled(bn_ds, sd, f"{ds_base}.1")
                    elif f"{ds_base}.1.weight" in sd:
                        copy_conv_scaled(
                            conv_ds.weight, sd[f"{ds_base}.1.weight"])
                        copy_bn_scaled(bn_ds, sd, f"{ds_base}.2")


if __name__ == "__main__":
    print("=" * 80)
    print("OClip (ResNet-vd) -> YOLOv11 Optimized Backbone Transfer")
    print("=" * 80)

    try:
        sd = load_oclip_state()
        print(f"Loaded {len(sd)} keys.")
        model = YOLO("yolov11n_seg.yaml")

        init_backbone_weights(model.model, sd)

        save_path = "yolov11n_oclip_transfer.pt"
        model.save(save_path)
        print(f"\nSaved transferred model to: {save_path}")
        print("Note: SE Blocks and extra Stem layers (if any) are randomly initialized.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CKPT_PATH}")
    except Exception as e:
        print(f"Error during transfer: {e}")
