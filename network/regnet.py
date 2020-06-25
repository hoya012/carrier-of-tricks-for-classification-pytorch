import torch.nn as nn
import numpy as np
from network.anynet import AnyNet

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont

class RegNet(AnyNet):
    """RegNetY-1.6GF model."""
    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='RegNet',):
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        

        SE_ON = True
        DEPTH = 27
        W0 = 48
        WA = 20.71
        WM = 2.65
        GROUP_W = 24

        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            w_a=WA, w_0=W0, w_m=WM, d=DEPTH
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [GROUP_W for _ in range(num_s)]
        bms = [1.0 for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25 if SE_ON else None
        # Construct the model
        kwargs = {
            "stem_type": "simple_stem_in",
            "stem_w": 32,
            "block_type": "res_bottleneck_block",
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": self.num_classes
        }
        super(RegNet, self).__init__(shape, num_classes, checkpoint_dir, checkpoint_name, **kwargs)