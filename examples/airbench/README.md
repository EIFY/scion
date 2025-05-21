# Scion on [airbench](https://github.com/KellerJordan/cifar10-airbench/tree/1da61ae58ee9c112e7f166fac3b97d245aa72942)

Scion simplifies the training setup by:

- avoiding Frobenius normalization within the optimizer
- remove the need for training the whiten bias (i.e., `whiten_bias_epoch=0`)

## Overview

- [`airbench_muon.py`](airbench_muon.py): 94.04% (mean over 200 runs)
- [`airbench_sgd.py`](airbench_sgd.py): 94.01% (mean over 200 runs)
- [`airbench_scion.py`](airbench_scion.py): 93.95% (mean over 50 runs)
- [`airbench_scion_speedrun.py`](airbench_scion_speedrun.py): 94.07% (mean over 200 runs) through further optimization of the scaling factors and the learning rate.


## Pseudocode

The configuration used in `airbench_scion.py`:

```python
radius = 8.0
optim_groups = [{
    'params': conv_layers,
    'norm': 'SpectralConv',
    'norm_kwargs': {'steps': 9}, # to stay consistent with the Muon baseline
    'scale': radius,
}, {
    'params': batchnorm_layers,
    'norm': 'BiasRMS', # heuristically uses l2 for normalization layers
    'norm_kwargs': {},
    'scale': radius,
}, {
    'params': output_layer,
    'norm': 'Sign',
    'norm_kwargs': {'normalized': True},
    'scale': radius*16,
}]
optimizer = Scion(optim_groups, lr=0.05, momentum=0.6)
```

The implementation uses the Newton-Schulz version used in the Muon baseline for fair comparison.
