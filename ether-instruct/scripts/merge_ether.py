# ------------------------------------------------------------------------------------------
# Implementation derived from: 
# ⚬ litgpt (https://github.com/Lightning-AI/litgpt), License: Apache License 2.0
# ------------------------------------------------------------------------------------------
"""This script merges the ETHER weights with the base model"""

import os, sys
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.ether import GPT, Config, ether_filter, merge_ether_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load


def merge_ether(
    ether_path: Path = Path(".logs/ether/alpaca/lit_model_ether_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: str = "",
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-ether model.
    See `finetune/ether.py`.

    Args:
        ether_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/ether.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        out_dir: The path to the merged model that is created by this script.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=1, precision=precision)

    check_valid_checkpoint_dir(checkpoint_dir)

    if out_dir == "":
        if ether_path.name == "lit_model_ether_finetuned.pth":
            out_dir = ether_path.parent / "merged_final"
        elif ether_path.name[:4] == "iter":
            iter = ether_path.name.split("-")[1]
            out_dir = ether_path.parent / f"merged_iter{iter}"
        else:
            out_dir = ether_path.parent / "merged"
    else:
        out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    settings = ether_path.parent.name.split("_")
    Htype = settings[1].split('-')[0]
    if settings[1].split('-')[1] == 'Hb':
        Gtype = 'Gsame'
    elif settings[1].split('-')[1] == 'Gb':
        Gtype = 'Gnew'
    else:
        Gtype = 'Gsnot'
    ether_r = int(settings[1].split('-r')[1].split('-')[0])
    qkvpmh = settings[2]
    ether_query = 'q' in qkvpmh
    ether_key = 'k' in qkvpmh
    ether_value = 'v' in qkvpmh
    ether_projection = 'p' in qkvpmh
    ether_mlp = 'm' in qkvpmh
    ether_head = 'h' in qkvpmh
    print(f"~~ settings: {settings}, Htype: {Htype}, Gtype: {Gtype}, ether_r: {ether_r}~~")

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=ether_r,
        Htype=Htype,
        Gtype=Gtype,
        dropout=0,
        to_query=ether_query,
        to_key=ether_key,
        to_value=ether_value,
        to_projection=ether_projection,
        to_mlp=ether_mlp,
        to_head=ether_head,
    )

    with fabric.init_module(empty_init=True):
        model = GPT(config)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint = lazy_load(checkpoint_path)
    ether_checkpoint = lazy_load(ether_path)
    checkpoint.update(ether_checkpoint.get("model", ether_checkpoint))

    model.load_state_dict(checkpoint)

    merge_ether_weights(model)

    save_path = out_dir / "lit_model.pth"
    fabric.print(f"Saving weights to {str(save_path)!r}")
    # remove ether parameters and the ether linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not ether_filter(k, v)}
    torch.save(state_dict, save_path)

    print("Merged model saved")
    # copy json files from ether checkpoint to new folder
    #os.chdir(out_dir)
    os.system(f"ln -s {checkpoint_dir.resolve()}/lit_config.json {out_dir.resolve()}")
    os.system(f"ln -s {checkpoint_dir.resolve()}/generation_config.json {out_dir.resolve()}")
    os.system(f"ln -s {checkpoint_dir.resolve()}/tokenizer_config.json {out_dir.resolve()}")
    try:
        os.system(f"ln -s {checkpoint_dir.resolve()}/tokenizer.json {out_dir.resolve()}")
    except:
        pass
    try:
        os.system(f"ln -s {checkpoint_dir.resolve()}/tokenizer.model {out_dir.resolve()}")
    except:
        pass
    print("Symlinks done")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(merge_ether)
