import json
import logging
import os
from argparse import ArgumentParser
from args import args
from pathlib import Path
from typing import Dict, List
import pickle

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model
import argparse

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


# def _conv_stack_cached(dilations, in_channels, out_channels, kernel_size):
#     return nn.ModuleList(
#         [
#             CausalConv1dCached(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 dilation=d,
#                 kernel_size=kernel_size,
#             )
#             for i, d in enumerate(dilations)
#         ]
#     )

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


# class CausalConv1dCached(nn.Module):
#     def __init__(
#         self, 
#         in_channels: int, 
#         out_channels: int, 
#         kernel_size: int, 
#         stride: int = 1, 
#         dilation: int = 1 , 
#         groups: int = 1, 
#         bias: bool = True) -> None:

#         super().__init__()
#         # self.padding = (kernel_size - 1) * dilation
#         self.padding = kernel_size // 2 * dilation
#         self.pad = PaddingCached(self.padding * 2, in_channels)
        
#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             # padding=self.padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )

#     def forward(self, input: Tensor) -> Tensor:
#         print(f'padding: {self.padding}')
#         x = self.pad(input)
#         result = self.conv(x)
#         if self.padding != 0:
#             return result[:, :, : -self.padding]
#         return result


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


class PedalNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PedalNet, self).__init__()
        self.wavenet = WaveNet(
            num_channels=hparams["num_channels"],
            dilation_depth=hparams["dilation_depth"],
            num_repeat=hparams["num_repeat"],
            kernel_size=hparams["kernel_size"],
        )
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(os.path.dirname(self.hparams.model) + "/data.pickle", "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4)

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean()
        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


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
        return True

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/ryok/guitar-amplifier_wavenet"
        }

    def get_citation(self) -> str:
        return ""

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return [44100]  # Supports all sample rates

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:

        # standardize of train_data
        # mean = 7.88423221820267e-06
        # std = 0.0439111590385437
        # x = (x - mean) / std
        
        # main process
        for ch in range(x.shape[0]):  # process channel by channel
            x_ = x[ch].reshape(1, 1, -1)
            x_ = self.model.wavenet(x_)
            x[ch] = x_.squeeze()
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args_main = parser.parse_args()
    root_dir = Path(args_main.output)

    # pedal load
    args.num_channels = 12
    args.dilation_depth = 10
    args.num_repeat = 1
    args.kernel_size = 3

    weights_path = "~/PedalNetRT/models/pedalnet/pedalnet3.ckpt"
    # weights_path = "~/guitar-amplifier_wavenet/pedalnet.ckpt"
    pedal = PedalNet(vars(args))
    # pedal.load_state_dict(
    #     torch.load(
    #         weights_path, 
    #         map_location=torch.device('cpu')
    #     )['state_dict'])
    model = PedalNet.load_from_checkpoint(weights_path)

    wrapper = WaveNetWrapper(model)
    metadata = wrapper.to_metadata()
    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )

    # Check model was converted correctly
    script, _ = load_neutone_model(root_dir / "model.nm")
    # log.info(script.set_daw_sample_rate_and_buffer_size(48000, 2048))
    log.info(script.set_daw_sample_rate_and_buffer_size(44100, 2048))
    log.info(script.reset())
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))
