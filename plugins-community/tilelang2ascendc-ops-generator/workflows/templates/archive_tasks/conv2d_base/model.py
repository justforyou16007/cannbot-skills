import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Conv2d reference model (CPU golden).

    forward(x, weight, bias=None, stride=[1,1], padding=[0,0], dilation=[1,1],
            data_format="NCHW", groups=1)

    Reference semantics:
      - All inputs are moved to CPU and cast to float32.
      - NHWC input is permuted to NCHW, computed with torch.nn.functional.conv2d,
        then permuted back to NHWC.
      - bias is optional.
      - groups: grouped/depthwise convolution (groups == C with weight
        [Co, 1, kH, kW] is depthwise; Co % groups == 0).
      - Result is cast back to x.dtype.

    Attribute conventions:
      - stride   = [stride_h, stride_w]
      - padding  = [pad_h, pad_w]  (F.conv2d padding=(pad_h, pad_w): first element pads H,
                                    second element pads W, matching the [h, w] spec directly)
      - dilation = [dilation_h, dilation_w]
      - data_format in {"NCHW", "NHWC"}; x logical shape is always [N, C, H, W]
        (NHWC tensors are stored physically as [N, H, W, C]).
      - weight shape is always [Co, Ci, kH, kW].
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias=None,
                stride=[1, 1], padding=[0, 0], dilation=[1, 1], data_format="NCHW",
                groups=1):
        out_dtype = x.dtype

        # NPU branch (performance benchmarking): native torch_npu conv2d,
        # keeping the input dtype. NHWC physical tensors go through the
        # logical-view permute (channels_last) and are converted back.
        if x.device.type != "cpu":
            if data_format == "NHWC":
                x_l = x.permute(0, 3, 1, 2)
                out = F.conv2d(x_l, weight, bias, stride=tuple(stride),
                               padding=tuple(padding), dilation=tuple(dilation),
                               groups=groups)
                return out.permute(0, 2, 3, 1).contiguous()
            return F.conv2d(x, weight, bias, stride=tuple(stride),
                            padding=tuple(padding), dilation=tuple(dilation),
                            groups=groups)

        # CPU branch (precision golden): fp32 domain
        x_f32 = x.detach().cpu().to(torch.float32)
        w_f32 = weight.detach().cpu().to(torch.float32)
        b_f32 = bias.detach().cpu().to(torch.float32) if bias is not None else None

        if data_format == "NHWC":
            # physical [N, H, W, C] -> logical [N, C, H, W]
            x_f32 = x_f32.permute(0, 3, 1, 2).contiguous()

        out = F.conv2d(x_f32, w_f32, b_f32,
                       stride=tuple(stride), padding=tuple(padding),
                       dilation=tuple(dilation), groups=groups)

        if data_format == "NHWC":
            # logical [N, Co, Ho, Wo] -> physical [N, Ho, Wo, Co]
            out = out.permute(0, 2, 3, 1).contiguous()

        return out.to(out_dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "conv2d_base.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x_info = None
        weight_info = None
        bias_info = None
        stride = [1, 1]
        padding = [0, 0]
        dilation = [1, 1]
        data_format = "NCHW"
        groups = 1

        for inp in inputs:
            name = inp.get("name", "")
            if name == "x":
                x_info = inp
            elif name == "weight":
                weight_info = inp
            elif name == "bias":
                bias_info = inp
            elif name == "stride":
                stride = list(inp["value"])
            elif name == "padding":
                padding = list(inp["value"])
            elif name == "dilation":
                dilation = list(inp["value"])
            elif name == "data_format":
                data_format = inp["value"]
            elif name == "groups":
                groups = int(inp["value"])

        dtype = dtype_map[x_info["dtype"]]

        # x: logical shape [N, C, H, W], uniform(-1, 1)
        x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-1.0, 1.0)
        if data_format == "NHWC":
            x = x.permute(0, 2, 3, 1).contiguous()

        # weight: [Co, Ci, kH, kW], uniform(-0.5, 0.5)
        weight = torch.empty(weight_info["shape"], dtype=dtype).uniform_(-0.5, 0.5)

        # bias (optional): [Co], uniform(-0.5, 0.5)
        bias = None
        if bias_info is not None:
            bias = torch.empty(bias_info["shape"], dtype=dtype).uniform_(-0.5, 0.5)

        input_groups.append([x, weight, bias, stride, padding, dilation, data_format, groups])

    return input_groups


def get_init_inputs():
    return []
