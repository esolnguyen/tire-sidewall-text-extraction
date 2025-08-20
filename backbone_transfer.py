import re
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import ResNetLayer, ResNetBlock
from ultralytics.nn.modules.conv import Conv

ckpt = torch.load('resnet50-oclip-7ba0c533.pth', map_location='cpu')
sd = ckpt.get('state_dict', ckpt)


def strip_prefix(k: str) -> str:
    for p in ('visual.', 'backbone.', 'model.visual.', 'module.visual.',
              'encoder.visual.', 'trunk.', 'img_encoder.'):
        if k.startswith(p):
            return k[len(p):]
    return k


sd = {strip_prefix(k): v for k, v in sd.items()}

yolo = YOLO('yolo_11.yml')
det = yolo.model

res_layers = [m for m in det.model.modules() if isinstance(m, ResNetLayer)]
assert len(
    res_layers) >= 4, "Backbone must contain ResNetLayer stages (check your YAML)."
yl1, yl2, yl3, yl4 = res_layers[-4:]


def yolo_blocks(y_layer: ResNetLayer):
    seen, out = set(), []
    for m in y_layer.modules():
        if isinstance(m, ResNetBlock) and id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def set_param(tensor, key):
    if key in sd and tensor.shape == sd[key].shape:
        tensor.copy_(sd[key])


def copy_block(stage_idx: int, block_idx: int, yb: ResNetBlock):
    base = f'layer{stage_idx}.{block_idx}'
    # cv1/bn1
    set_param(yb.cv1.conv.weight, f'{base}.conv1.weight')
    for a in ('weight', 'bias', 'running_mean', 'running_var'):
        set_param(getattr(yb.cv1.bn, a), f'{base}.bn1.{a}')
    # cv2/bn2
    set_param(yb.cv2.conv.weight, f'{base}.conv2.weight')
    for a in ('weight', 'bias', 'running_mean', 'running_var'):
        set_param(getattr(yb.cv2.bn, a), f'{base}.bn2.{a}')
    # cv3/bn3
    set_param(yb.cv3.conv.weight, f'{base}.conv3.weight')
    for a in ('weight', 'bias', 'running_mean', 'running_var'):
        set_param(getattr(yb.cv3.bn, a), f'{base}.bn3.{a}')
    # downsample (1x1 + BN), if present
    ds = getattr(yb, 'downsample', None)
    if isinstance(ds, Conv):
        set_param(ds.conv.weight, f'{base}.downsample.0.weight')
        for a in ('weight', 'bias', 'running_mean', 'running_var'):
            set_param(getattr(ds.bn, a), f'{base}.downsample.1.{a}')
    elif isinstance(ds, nn.Sequential):
        conv = next((m for m in ds.modules()
                    if isinstance(m, nn.Conv2d)), None)
        bn = next((m for m in ds.modules() if isinstance(m, nn.BatchNorm2d)), None)
        if conv is not None:
            set_param(conv.weight, f'{base}.downsample.0.weight')
        if bn is not None:
            for a in ('weight', 'bias', 'running_mean', 'running_var'):
                set_param(getattr(bn, a), f'{base}.downsample.1.{a}')


with torch.no_grad():
    for stage_idx, y_layer in enumerate([yl1, yl2, yl3, yl4], start=1):
        y_blocks = yolo_blocks(y_layer)
        clip_block_ids = sorted({
            int(m.group(1)) for k in sd.keys()
            if k.startswith(f'layer{stage_idx}.') and (m := re.match(rf'layer{stage_idx}\.(\d+)\.', k))
        })
        assert len(y_blocks) <= len(clip_block_ids), \
            f'CLIP checkpoint has fewer blocks than your YOLO stage{stage_idx}'
        for i, yb in enumerate(y_blocks):
            copy_block(stage_idx, i, yb)


def _strip_prefix(k: str) -> str:
    for p in ('visual.', 'backbone.', 'model.visual.', 'module.visual.',
              'encoder.visual.', 'trunk.', 'img_encoder.'):
        if k.startswith(p):
            return k[len(p):]
    return k


def verify_transfer(src_sd: dict, yolo_obj, verbose_first=20, atol=1e-7):
    """Return dict(stats) and print a brief report."""
    sd = {_strip_prefix(k): v for k, v in src_sd.items()}
    det = yolo_obj.model
    y_layers = [m for m in det.model.modules() if isinstance(m,
                                                             ResNetLayer)][-4:]

    def yolo_blocks(y_layer: ResNetLayer):
        seen, out = set(), []
        for m in y_layer.modules():
            if isinstance(m, ResNetBlock) and id(m) not in seen:
                seen.add(id(m))
                out.append(m)
        return out

    def allclose(a, b):
        return a.shape == b.shape and torch.allclose(a.cpu(), b.cpu(), atol=atol, rtol=0)

    total = matched = missing = 0
    details = []

    for stage_idx, y_layer in enumerate(y_layers, start=1):
        blocks = yolo_blocks(y_layer)
        for b_idx, yb in enumerate(blocks):
            base = f'layer{stage_idx}.{b_idx}'
            pairs = [
                (yb.cv1.conv.weight, f'{base}.conv1.weight'),
                (yb.cv1.bn.weight,   f'{base}.bn1.weight'),
                (yb.cv1.bn.bias,     f'{base}.bn1.bias'),
                (yb.cv1.bn.running_mean, f'{base}.bn1.running_mean'),
                (yb.cv1.bn.running_var,  f'{base}.bn1.running_var'),

                (yb.cv2.conv.weight, f'{base}.conv2.weight'),
                (yb.cv2.bn.weight,   f'{base}.bn2.weight'),
                (yb.cv2.bn.bias,     f'{base}.bn2.bias'),
                (yb.cv2.bn.running_mean, f'{base}.bn2.running_mean'),
                (yb.cv2.bn.running_var,  f'{base}.bn2.running_var'),

                (yb.cv3.conv.weight, f'{base}.conv3.weight'),
                (yb.cv3.bn.weight,   f'{base}.bn3.weight'),
                (yb.cv3.bn.bias,     f'{base}.bn3.bias'),
                (yb.cv3.bn.running_mean, f'{base}.bn3.running_mean'),
                (yb.cv3.bn.running_var,  f'{base}.bn3.running_var'),
            ]

            ds = getattr(yb, 'downsample', None)
            if isinstance(ds, Conv):
                pairs += [
                    (ds.conv.weight, f'{base}.downsample.0.weight'),
                    (ds.bn.weight,   f'{base}.downsample.1.weight'),
                    (ds.bn.bias,     f'{base}.downsample.1.bias'),
                    (ds.bn.running_mean, f'{base}.downsample.1.running_mean'),
                    (ds.bn.running_var,  f'{base}.downsample.1.running_var'),
                ]
            elif isinstance(ds, nn.Sequential):
                conv = next((m for m in ds.modules()
                            if isinstance(m, nn.Conv2d)), None)
                bn = next((m for m in ds.modules()
                          if isinstance(m, nn.BatchNorm2d)), None)
                if conv is not None:
                    pairs += [(conv.weight, f'{base}.downsample.0.weight')]
                if bn is not None:
                    pairs += [
                        (bn.weight, f'{base}.downsample.1.weight'),
                        (bn.bias,   f'{base}.downsample.1.bias'),
                        (bn.running_mean, f'{base}.downsample.1.running_mean'),
                        (bn.running_var,  f'{base}.downsample.1.running_var'),
                    ]

            for tensor, key in pairs:
                total += 1
                if key not in sd:
                    missing += 1
                    if len(details) < verbose_first:
                        details.append(f'[MISSING] {key}')
                else:
                    ok = allclose(tensor.data, sd[key])
                    matched += int(ok)
                    if not ok and len(details) < verbose_first:
                        diff = (tensor.data.cpu() -
                                sd[key].cpu()).abs().max().item()
                        details.append(
                            f'[DIFF] {key} max|Δ|={diff:.3e} y={tuple(tensor.shape)} s={tuple(sd[key].shape)}')

    kept = total - missing
    ratio = (matched / kept) if kept > 0 else 0.0
    print(f'Verify: matched {matched}/{kept} ({ratio:.2%}) among present tensors; '
          f'missing {missing} of {total} total checked.')
    for line in details:
        print(' ', line)
    return {'total': total, 'kept': kept, 'matched': matched, 'missing': missing, 'ratio_present': ratio}


# 1) Verify NOW (right after transfer)
stats_before_save = verify_transfer(sd, yolo)
assert stats_before_save['ratio_present'] > 0.99, "Too many mismatches — check mapping."

# 2) Save a portable snapshot (CPU)
snap_path = 'yolov11_resnet50_init.pth'
cpu_state = {k: v.detach().cpu() for k, v in det.state_dict().items()}
torch.save(cpu_state, snap_path)
print('Saved:', snap_path)

# 3) Reload into a fresh model and verify AGAIN (ensures the file is correct)
y_chk = YOLO('yolo_11.yml')
y_chk.model.load_state_dict(torch.load(
    snap_path, map_location='cpu'), strict=False)
stats_after_load = verify_transfer(sd, y_chk)
assert stats_after_load['matched'] == stats_before_save['matched'] and \
    stats_after_load['kept'] == stats_before_save['kept'], "Reloaded weights differ from pre-save."

# 4) Quick forward smoke test (no NaNs/Infs, correct shapes)
y_chk.model.eval()
with torch.no_grad():
    _ = y_chk.model(torch.randn(1, 3, 640, 640))
print('Forward pass OK.')
