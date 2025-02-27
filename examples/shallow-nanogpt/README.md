# Shallow NanoGPT

See [`mup_examples/README.md`](mup_examples/README.md) for setup.

```bash
# Transfer lr/width sweep
bash mup_examples/mutransfer_lr_shakespeare_char/uscion/run.sh
bash mup_examples/mutransfer_lr_shakespeare_char/scion/run.sh
bash mup_examples/mutransfer_lr_shakespeare_char/scion_full/run.sh
bash mup_examples/mutransfer_lr_shakespeare_char/sp/run.sh
bash mup_examples/mutransfer_lr_shakespeare_char/mup/run.sh

# Testing a single run (smallest width with optimal stepsize)
bash mup_examples/mutransfer_lr_shakespeare_char/uscion/run_test.sh
bash mup_examples/mutransfer_lr_shakespeare_char/scion/run_test.sh
bash mup_examples/mutransfer_lr_shakespeare_char/scion_full/run_test.sh
bash mup_examples/mutransfer_lr_shakespeare_char/sp/run_test.sh
bash mup_examples/mutransfer_lr_shakespeare_char/mup/run_test.sh
```


## CHANGELOG

Changes made (see `SCION_CHANGE` code comments):

- Modernized architecture:
    - Rotary embedding
    - RMS norm instead of LayerNorm
    - No weight sharing for first and last layer
    - Linear decay instead of cosine
    - GELU `sqrt(2)` scaling
- Decouples QKV into three separate `Linear` layers to expose each independently to the optimizer
- Increases batch size to 32 (maximum allowed for 4096 width model on an A100)
- Disables `torch.compile` to support running without triton
- Logs final validation loss (instead of a running average)


## Acknowledgements 

This codebase builds on [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup/).
See [`README_orig.md`](README_orig.md) for the original readme.
