## Language Models Finetuning

We report here the code for reproducing results for Instruction Tuning experiments. We report a minimal version of the code which mainly involves experiments shown in the paper. 
>For using the full [litgpt](https://github.com/Lightning-AI/litgpt.git) library (with different pretrained models, finetuning methods, tests, ...), we recommend cloning the repository (`git clone https://github.com/Lightning-AI/litgpt.git`) and revert it to the last tested compatible commit (`git checkout f241d94df59d82b2017bfdcd3800ac8779eb45f5`).

### Setup (with ⚡ Lit-GPT)

Create conda environment
```bash
conda create -n ether-instruct -c conda-forge python=3.10

conda activate ether-instruct
```

Install dependencies from litgpt:

```bash
pip install -r requirements.txt
```

Download a model checkpoint (e.g. Llama2-7B) (check how to select different models in [#Options](#Options))
```bash
python scripts/download.py \
    --repo_id meta-llama/Llama-2-7b-hf \
    --from_safetensors true \
    --access_token=<HF_TOKEN>
```

Convert to litgpt checkpoint
```bash
python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

Prepare finetuning dataset (e.g. Alpaca)
```bash
python scripts/prepare_alpaca.py \
    --destination_path checkpoints/meta-llama/Llama-2-7b-hf/alpaca-t256 \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
    --max_seq_length 256  # optional: we truncate at 256 to make it fit on a RTX3090
```

Install evaluation dependencies from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
```bash 
pip install https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/heads/master.zip -U
```

### Finetune and Evaluate

Finetune with _ETHER_, _ETHER+_
```bash
python finetune/ether.py \
    --io.train_data_dir checkpoints/meta-llama/Llama-2-7b-hf/alpaca-t256 \
    --io.log_dir logs \
    --train.learning_rate 1e-3 \
    --train.global_batch_size 16 \
    --train.micro_batch_size 1 \
    --train.epochs 1 \
    --train.epoch_size 50000 \
    --precision bf16-true \
    --ether_dropout 0.0 \
    --ether_nb 32 \
    --Htype etherplusHH   # ether, etherplusHH, etherplus (one-sided version of ETHER+), oft
```

Merge Finetuned Weights
```bash
python scripts/merge_ether.py \
    --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
    --precision bf16-true \
    --ether_path path/to/lit_model_ether_finetuned.pth
```
                
Test Generation
```bash
python chat/base.py \ 
    --checkpoint_dir path/to/merged_model/
```


Evaluate
```bash
python eval/lm_eval_harness.py \
    --eval_tasks "['arc_challenge', 'truthfulqa_mc', 'hendrycksTest-*']" \
    --precision bf16-true \
    --checkpoint_dir path/to/merged_model/
```

### Options

Choose pretrained model:
- to show available pretrained models (e.g. from Meta) do
```bash
python scripts/download.py | grep meta

# Output
# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Llama-2-13b-hf
# meta-llama/Llama-2-13b-chat-hf
# meta-llama/Llama-2-70b-hf
# meta-llama/Llama-2-70b-chat-hf
```

Choose multiplicative finetuning method by changing `--Htype` in `finetune/ether.py`:
- `ether`: _ETHER_
- `etherplus`: _ETHER+_ (one-sided)
- `etherplusHH`: _ETHER+_
- `oft`: OFT
- to run LoRA, use `finetune/lora.py` instead

Choose number of diagonal blocks (see [#Best Practices](../#Best-Practices) ):
- `--ether_nb 32` in `finetune/ether.py`

Apply (one-sided) finetuning methods on smaller side (to non-square pretrained weights):
- `--flip_side True`


### Note

- Due to Q,K,V interleaving in LitGPT implementation (as pointed out in [issue #891](https://github.com/Lightning-AI/litgpt/issues/891)), we implemented _ETHER_ finetuning by indexing the desired Q,K,V matrices. However this is suboptimal w.r.t. simply injecting/concatenating identity matrices along with _ETHER_ matrices and then running the matrix-multiplication. We expect this latter implementation to lead to a significant speed-up, so we recommend following this if re-implementing _ETHER_ on a different codebase.

- While in most of our experiments we do not employ regular dropout, [Liu et al.](https://arxiv.org/abs/2311.06243) propose a _multiplicative dropout_ form specifically designed for multiplicative finetuning methods, which we did not implement or test in this study, but we expect to work better than regular dropout.
