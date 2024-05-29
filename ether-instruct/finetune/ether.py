# ------------------------------------------------------------------------------------------
# Implementation derived from: 
# ⚬ litgpt (https://github.com/Lightning-AI/litgpt), License: Apache License 2.0
# ⚬ oft (https://github.com/Zeju1997/oft), License: MIT License
# ------------------------------------------------------------------------------------------

import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.args import EvalArgs, IOArgs, TrainArgs
from lit_gpt.ether import GPT, Block, Config, ether_filter, mark_only_ether_as_trainable
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    CLI,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt
import datetime

eval_interval = 100
save_interval = 100
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 1
devices = 1


def setup(
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: int = 1,
    seed: int = 1337,
    sample_with_replacement: bool = False, # set True to revert to default litgpt implementation
    flip_side: bool = False, # set True to apply finetuning on smaller side (when non-squared)
    # ether model
    Htype: str = 'ether',
    ether_nb: int = 1,
    ether_dropout: float = 0.0,
    # ether layers
    ether_query: bool = True,
    ether_key: bool = False,
    ether_value: bool = True,
    ether_projection: bool = False,
    ether_mlp: bool = False,
    ether_head: bool = False,
    io: IOArgs = IOArgs(
        train_data_dir=Path("checkpoints/microsoft/phi-1_5/alpaca"),
        val_data_dir=None,
        checkpoint_dir=None,
        log_dir=Path(".logs"),
        jobnum="",
    ),
    train: TrainArgs = TrainArgs(
        weight_decay=0.0,
        lr_warmup_steps=1000,
        max_seq_length=None,
        save_interval=1000,
        log_interval=1,
        # to tune
        epochs=2,
        epoch_size=50000,
        learning_rate=3e-4,
        global_batch_size=128,
        micro_batch_size=4,
    ),
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
) -> None:
    print(locals())

    if io.checkpoint_dir is None:
        io.checkpoint_dir = io.train_data_dir.parent

    ether_config=Config.from_name(
        name=io.checkpoint_dir.name,
        Htype=Htype,
        nb=ether_nb,
        flip_side=flip_side,
        dropout=ether_dropout,
        to_query=ether_query,
        to_key=ether_key,
        to_value=ether_value,
        to_projection=ether_projection,
        to_mlp=ether_mlp,
        to_head=ether_head,
    )

    settings = f'{Htype}-'
    settings += f'r{ether_nb}_'
    settings += ether_query*'q' + ether_key*'k' + ether_value*'v' + ether_projection*'p' + ether_mlp*'m' + ether_head*'h' + '_'
    settings += f'lr{format(train.learning_rate, ".0e")}_'.replace('0','')
    settings += f'bs{train.global_batch_size}-{train.micro_batch_size}_ep{train.epochs}-{train.epoch_size//1000}k_dr{ether_dropout}_wd{train.weight_decay}_wu{train.lr_warmup_steps}'
    if train.max_seq_length is not None:
        settings += f'_sq{train.max_seq_length}'

    out_dir_ckpt = io.checkpoint_dir.name.lower().replace('_','-')
    out_dir_ft = io.train_data_dir.name
    if io.jobnum != '':
        out_dir_model = f'j{io.jobnum}_' + settings
    else:
        now = datetime.datetime.now()
        dt_string = now.strftime("%m%d-%H%M%S")
        out_dir_model = f'd{dt_string}_' + settings
    out_dir_model += ('_' + precision.replace('-','')) if precision is not None else ''
    if sample_with_replacement:
        out_dir_model += '_SWR'
    if flip_side:
        out_dir_model += '_FS'
    io.out_dir = io.log_dir / out_dir_ckpt / out_dir_ft / out_dir_model
    # makedir
    os.makedirs(io.log_dir, exist_ok=True)
    os.makedirs(io.log_dir / out_dir_ckpt, exist_ok=True)
    os.makedirs(io.log_dir / out_dir_ckpt / out_dir_ft, exist_ok=True)
    os.makedirs(io.out_dir, exist_ok=True)

    precision = precision or get_default_supported_precision(training=True)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(io.out_dir.parent, io.out_dir.name, flush_logs_every_n_steps=train.log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)

    if not any((ether_query, ether_key, ether_value, ether_projection, ether_mlp, ether_head)):
        fabric.print("Warning: all ETHER layers are disabled!")
    fabric.launch(
        main,
        devices,
        seed,
        ether_config,
        io,
        train,
        eval,
        sample_with_replacement,
    )


def main(fabric: L.Fabric, devices: int, seed: int, config: Config, io: IOArgs, train: TrainArgs, eval: EvalArgs, sample_with_replacement) -> None:
    validate_args(io, train, eval)

    steps_per_epoch = train.epoch_size // devices // train.batch_size(devices)
    lr_max_steps = train.epochs * steps_per_epoch

    check_valid_checkpoint_dir(io.checkpoint_dir)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    train_data = torch.load(io.train_data_dir / "train.pt")
    if io.val_data_dir is not None:
        val_data = torch.load(io.val_data_dir / "test.pt")
    else:
        val_data = torch.load(io.train_data_dir / "test.pt")

    checkpoint_path = io.checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)
    mark_only_ether_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.PagedAdamW
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params, lr=train.learning_rate, weight_decay=train.weight_decay, betas=(train.beta1, train.beta2)
    )

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps, global_batch_size=train.global_batch_size)

    # strict=False because missing keys due to ETHER weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    fit(fabric, model, optimizer, scheduler, train_data, val_data, devices, io, train, eval, sample_with_replacement)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final ETHER checkpoint at the end of training
    save_path = io.out_dir / "lit_model_ether_finetuned.pth"
    save_ether_checkpoint(fabric, model, save_path)


def fit(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    devices: int,
    io: IOArgs,
    train: TrainArgs,
    eval: EvalArgs,
    sample_with_replacement,
) -> None:
    tokenizer = Tokenizer(io.checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, tokenizer, dataclasses.replace(eval, max_iters=2), train)  # sanity check


    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    train_data_popped = []
    for iter_num in range(1, train.max_iters(devices) + 1):
        iter_t0 = time.perf_counter()

        if not sample_with_replacement:
            if len(train_data_popped) < train.micro_batch_size:
                train_data_popped = train_data.copy()

            input_ids, targets = get_batch(
                fabric, train_data_popped, train.micro_batch_size, train.max_seq_length, longest_seq_ix if iter_num == 1 else None
            )
        else: # default litgpt implementation
            input_ids, targets = get_batch(
                fabric, train_data, train.micro_batch_size, train.max_seq_length, longest_seq_ix if iter_num == 1 else None, pop=False
            )

        is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1


        total_lengths += input_ids.numel()
        if iter_num % train.log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * train.micro_batch_size, lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} | step {step_count}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f} ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, eval, train)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            fabric.barrier()

        if not is_accumulating and step_count % train.save_interval == 0:
            checkpoint_path = io.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_ether_checkpoint(fabric, model, checkpoint_path)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, eval: EvalArgs, train: TrainArgs
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval.max_iters)
    for k in range(eval.max_iters):
        input_ids, targets = get_batch(fabric, val_data, train.micro_batch_size, train.max_seq_length, pop=False)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model, encoded, max_returned_tokens=len(encoded) + eval.max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric,
    data: List[Dict],
    micro_batch_size: int,
    max_seq_length: Optional[int],
    longest_seq_ix: Optional[int] = None,
    pop: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if pop:
        input_ids_labels = []
        for _ in range(micro_batch_size):
            if longest_seq_ix is not None:
                ix = longest_seq_ix # force the longest sample at the beginning so potential OOMs happen right away
                longest_seq_ix = None
            else:
                ix = torch.randint(len(data), (1,))
            input_ids_labels.append(data.pop(ix))
        input_ids = [x["input_ids"].type(torch.int64) for x in input_ids_labels]
        labels = [x["labels"].type(torch.int64) for x in input_ids_labels]
    else:
        ix = torch.randint(len(data), (micro_batch_size,))
        input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
        labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n_pad = max_len - len(x)
        return torch.cat((x, torch.full((n_pad,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, global_batch_size: int = 32):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_ether_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving ETHER weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": ether_filter})


def validate_args(io: IOArgs, train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [
        (io, ["checkpoint_dir", "train_data_dir"]), #~, "val_data_dir"
        (train, ["epoch_size", "epochs"]),
        (eval, ["max_new_tokens"]),
    ]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
