# Experiment Reproduction

Install the minimal dataset and plotting requirements with `pip install -r requirements.txt`. We used the PyTorch NGC container for GPU-based runs, but any environment containing the dependencies from [the main README](https://github.com/EleutherAI/nanoGPT-mup?tab=readme-ov-file#install) will suffice.

SCION_CHANGE: install `sudo apt install bc`.
SCION_CHANGE: install `pip install tiktoken`

To download the tiny shakespeare dataset, run `python ../data/shakespeare_char/prepare.py`. For OpenWebText (OWT), run `python ../data/openwebtext/prepare.py`.
