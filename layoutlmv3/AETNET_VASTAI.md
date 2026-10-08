# AETNet on Vast.ai

## Architecture and scope

The model is an opt-in token-classification extension of this fork's LayoutLMv3. The original pretrained multimodal transformer remains the backbone and its ordinary behavior is unchanged unless `--use_aetnet` is selected.

```text
Document image -> LayoutLMv3 patch embedding + gated CNN side features --+
                                                                       |
OCR tokens -> LayoutLMv3 text/layout embedding + learned soft hints ----+-> LayoutLMv3 multimodal encoder
                                                                          -> gated text-to-patch attention
                                                                          -> token classifier
                                                                          -> supervised loss
                                                                             + DITC + IMC + GLITC + PITA
```

The CNN side encoder makes ImageEnhance spatial (rather than globally pooled), so its output aligns with image patches. Soft prompts borrow the PrecisionHints/P-Tuning idea but do not fabricate absent receipt fields. Cross-modal attention extends LayoutLMv3's existing fusion. The alignment loss adapts the AET paper: symmetric in-batch document contrast, dropout-based two-view intra-modal contrast, global/local cross-modal contrast, and OCR-box-to-patch cosine alignment. It uses no momentum queue or separate full ViT/RoBERTa towers, trading fidelity to the paper for a smaller, practical fine-tuning model. Alignment loss is training-only; evaluation reports the standard supervised task metrics.

## Rent and connect a GPU

1. Create/sign in to a Vast.ai account and add billing credit. Avoid sharing API keys or private SSH keys in chat or in this repository.
2. In the GPU marketplace, filter for an Ubuntu/PyTorch-capable machine with a CUDA GPU. Prefer at least 24 GB VRAM for LayoutLMv3-base; 40 GB or more gives more room. A 16 GB card may work with batch size 1, but the in-batch contrastive terms then have no negative examples. Gradient accumulation does not increase the contrastive batch; keep per-device batch size at 2 or higher for the intended combined objective.
3. Compare GPU VRAM, host reliability, disk space, bandwidth, hourly price, and location. Start with an on-demand rental for a short smoke test; check the current instance terms and storage charges before confirming.
4. Select a PyTorch/CUDA template, allocate persistent disk, and expose SSH access. Start the instance and copy the SSH command shown by Vast.ai. Connect from your local terminal with that command; accept the host key only if the displayed host/fingerprint matches the instance details.
5. Keep the repository and datasets on persistent storage, not temporary instance storage. Stop or destroy the instance when finished; confirm what happens to disk storage separately because persistent disks can continue to incur charges.

The marketplace UI and available templates change, so use the current Vast.ai instance panel for the exact rental, SSH, and port-forwarding values.

## Prepare the instance

After connecting over SSH:

```bash
git clone <YOUR_GITHUB_FORK_URL> unilm
cd unilm/layoutlmv3
python --version
nvidia-smi
```

Use the Python/PyTorch/CUDA combination supported by the selected template. This repository pins an older stack (`transformers==4.12.5`, `timm==0.4.12`); do not blindly upgrade Transformers, because this fork uses older model APIs. If the template does not include PyTorch, first install the official PyTorch/torchvision wheel matching its CUDA runtime, using the current selector at pytorch.org. Do that before installing `requirements.txt`; otherwise `timm` may cause pip to select an unintended torch build. Then prepare the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

Verify `torch.cuda.is_available()` is `True` before paying for a training run. This fork's requirements file does not pin torch; the original README documents an older torch 1.10/CUDA 11.1 combination.

FUNSD and CORD are loaded by the repository's dataset scripts. For CORD, first confirm that the dataset script can access the dataset in the instance environment. Private Finance-Receipts data from the DocExtractNet paper is not included here and must be provided by its owner.

## Train and evaluate FUNSD

First run a short smoke test to catch dependency, dataset, and GPU problems:

```bash
python examples/run_funsd_cord.py \
  --dataset_name funsd --do_train --do_eval --use_aetnet \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir output/aetnet-funsd-smoke \
  --input_size 224 --max_steps 5 --save_steps 5 \
  --learning_rate 1e-5 --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 \
  --fp16 --dataloader_num_workers 2
```

After the smoke test succeeds, start the full run:

```bash
python examples/run_funsd_cord.py \
  --dataset_name funsd --do_train --do_eval --use_aetnet \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir output/aetnet-funsd \
  --input_size 224 --max_steps 1000 --save_steps 250 \
  --evaluation_strategy steps --eval_steps 250 \
  --learning_rate 1e-5 --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 --gradient_accumulation_steps 2 \
  --fp16 --dataloader_num_workers 4
```

Evaluate from the saved model (keep `--use_aetnet` enabled so the extra layers are reconstructed):

```bash
python examples/run_funsd_cord.py \
  --dataset_name funsd --do_eval --use_aetnet \
  --model_name_or_path output/aetnet-funsd \
  --output_dir output/aetnet-funsd-eval \
  --input_size 224 --per_device_eval_batch_size 2 \
  --fp16 --dataloader_num_workers 4
```

Use `--dataset_name cord` for CORD. The script evaluates the dataset's `test` split under `--do_eval` and saves standard `seqeval` precision, recall, F1, and accuracy metrics under the output directory. Report multiple seeds and compare against the baseline LayoutLMv3 run using the same data, preprocessing, and training budget; this fork has not yet validated the paper's published scores.

## Fix `Invalid URL` from Hugging Face

With the pinned `transformers==4.12.5`, Hugging Face may return a relative redirect such as `/api/resolve-cache/...`. This Transformers version uses that `Location` value as-is instead of resolving it against `https://huggingface.co`, which makes `requests` fail with `No scheme supplied`. This is a downloader compatibility issue, not an invalid model ID.

Activate the same virtual environment used for training, then apply this guarded patch to its installed Transformers file. It creates a `.bak` backup and refuses to modify the file if the expected old code is not present:

```bash
python - <<'PY'
from pathlib import Path
import shutil
import transformers.file_utils as file_utils

path = Path(file_utils.__file__)
source = path.read_text()
old_import = "from urllib.parse import urlparse"
new_import = "from urllib.parse import urljoin, urlparse"
old_redirect = 'url_to_download = r.headers["Location"]'
new_redirect = 'url_to_download = urljoin(url, r.headers["Location"])'

if old_redirect not in source:
  raise SystemExit("Expected Transformers 4.12.5 redirect code was not found; no changes made.")
if old_import not in source:
  raise SystemExit("Expected urllib.parse import was not found; no changes made.")

backup = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, backup)
source = source.replace(old_import, new_import, 1).replace(old_redirect, new_redirect, 1)
path.write_text(source)
print("Patched:", path)
print("Backup:", backup)
PY
```

Verify that config download now works, then retry training:

```bash
python -c 'from transformers import AutoConfig; print(AutoConfig.from_pretrained("microsoft/layoutlmv3-base").model_type)'
```

The expected output is `layoutlmv3`. This patch is inside the virtual environment and will need to be reapplied if Transformers is reinstalled. Do not apply it if the guard reports that the expected code was not found; inspect the installed Transformers version/source first.

## Keep results and control cost

- Use `tmux` or `screen` for SSH sessions that may disconnect. Save logs and checkpoints under the persistent disk.
- Check `nvidia-smi` and the process log during the smoke test. If CUDA runs out of memory, lower batch size first, then reduce `--input_size` consistently (it must be divisible by 16); increase gradient accumulation to preserve the effective batch size.
- Copy checkpoints and metrics to your own durable storage before terminating the instance. Confirm the transfer completed, then stop the GPU instance and review the disk's billing/retention state.
- Never put Vast.ai credentials, SSH private keys, private dataset files, or billing details into Git history.