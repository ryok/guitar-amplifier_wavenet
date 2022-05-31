import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class CausalConv1d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        dilation: int = 1 , 
        groups: int = 1, 
        bias: bool = True) -> None:

        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        result = self.conv(input)
        if self.padding != 0:
            return result[:, :, : -self.padding]
        return result


class WaveNet(nn.Module):
    def __init__(
        self, num_channels: int = 3, 
        dilation_depth: int = 2, 
        num_repeat: int = 2, 
        kernel_size: int = 2
        ) -> None:
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )
        self.num_channels = num_channels

        self.initialize_random()

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out

    def weights_init(self, m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname == 'Linear':
            nn.init.normal_(m.weight, 0, 0.40)

    def initialize_random(self) -> None:
        for n in self.hidden:
            nn.init.normal_(n.conv.weight, 0, 0.7)
        for n in self.residuals:
            nn.init.normal_(n.conv.weight, 0, 0.7)
        nn.init.normal_(self.input_layer.conv.weight, 0, 0.7)
        nn.init.normal_(self.linear_mix.weight, 0, 0.7)


class WaveNetWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "wavenet-amplifier.random"

    def get_model_authors(self) -> List[str]:
        return ["Ryo Okada"]

    def get_model_short_description(self) -> str:
        return "Neural guitar amplifier effect"

    def get_model_long_description(self) -> str:
        return "Neural guitar amplifier effect through randomly initialized WaveNet"

    def get_technical_description(self) -> str:
        return "Random guitar amplifier effect through randomly initialized WaveNet. Based on the idea proposed by Alec Wright."

    def get_tags(self) -> List[str]:
        return ["guitar", "amplifier"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/GuitarML/PedalNetRT"
        }

    def get_citation(self) -> str:
        return ""

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # main process
        for ch in range(x.shape[0]):  # process channel by channel
            x_ = x[ch].reshape(1, 1, -1)
            x_ = self.model(x_)
            x[ch] = x_.squeeze()
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = Path(args.output)

    model = WaveNet()
    wrapper = WaveNetWrapper(model)
    metadata = wrapper.to_metadata()
    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )

    # Check model was converted correctly
    script, _ = load_neutone_model(root_dir / "model.nm")
    log.info(script.set_daw_sample_rate_and_buffer_size(48000, 2048))
    log.info(script.reset())
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))
