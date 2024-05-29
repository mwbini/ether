# ------------------------------------------------------------------------------------------
# Implementation derived from: 
# ⚬ litgpt (https://github.com/Lightning-AI/litgpt), License: Apache License 2.0
# ------------------------------------------------------------------------------------------
"""This script merges the LoRA weights with the base model"""

import sys
from pathlib import Path
from typing import Optional
import os

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.lora import GPT, Config, lora_filter, merge_lora_weights
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, lazy_load


def merge_lora(
    lora_path: Path = Path(".logs/alpaca/lora/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = "",
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        out_dir: The path to the merged model that is created by this script.
        precision: Indicates the Fabric precision setting to use.
    """
   
    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=1, precision=precision)
    check_valid_checkpoint_dir(checkpoint_dir)

    if out_dir.name == "":
        if lora_path.name == "lit_model_lora_finetuned.pth":
            out_dir = lora_path.parent / "merged_final"
        elif lora_path.name[:4] == "iter":
            iter = lora_path.name.split("-")[1]
            out_dir = lora_path.parent / f"merged_iter{iter}"
        else:
            out_dir = lora_path.parent / "merged"
    else:
        out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)


    settings = lora_path.parent.name.split("_")
    lora_r = int(settings[1].split('-r')[1].split('-')[0])
    lora_alpha = int(settings[1].split('-a')[1].split('-')[0])
    qkvpmh = settings[2]
    lora_query = 'q' in qkvpmh
    lora_key = 'k' in qkvpmh
    lora_value = 'v' in qkvpmh
    lora_projection = 'p' in qkvpmh
    lora_mlp = 'm' in qkvpmh
    lora_head = 'h' in qkvpmh
    print(f"~~ settings: {settings}, lora_r: {lora_r}, lora_alpha: {lora_alpha} ~~")


    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=0,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    with fabric.init_module(empty_init=True):
        model = GPT(config)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)

    merge_lora_weights(model)

    save_path = out_dir / "lit_model.pth"
    fabric.print(f"Saving weights to {str(save_path)!r}")
    # remove lora parameters and the lora linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    torch.save(state_dict, save_path)

    # copy json files from lora checkpoint to new folder
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

if __name__ == "__main__":
    CLI(merge_lora)
