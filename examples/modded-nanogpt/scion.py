import itertools
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np

#######################################################
# Scion
#######################################################
eps = 1e-8


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError

    def momentum_buffer_shape(self, w):
        return w.shape

    def weight_cosine_shape(self, w):
        return ()

    def singular_shape(self, w):
        raise NotImplementedError

    def singular_cosine_shape(self, w):
        return self.singular_shape(w)[:-2]


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def scale(self, d_out, d_in):
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        return scale

    def lmo(self, g):
        g = PolarExpress(g, steps=self.steps)
        g *= self.scale(*g.shape[-2:])
        return g

    def dual_norm(self, g):
        return torch.sum(self.lmo(g) * g, dim=(-2, -1), keepdim=True)

    @torch.compile
    def norm(self, w, v, repeat=1):
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            v = w.mT @ u
            s = torch.linalg.vector_norm(v, dim=-2, keepdim=True)
            v /= s
        scale = self.scale(*w.shape[-2:])
        return s / scale, v

    def init(self, w, init_dtype=torch.float64):
        v_shape = w.shape[:-2] + (w.shape[-1], 1)
        v = torch.normal(0, 1, v_shape).to(w)
        v /= torch.linalg.vector_norm(v, dim=-2, keepdim=True)
        return self.norm(w, v)

    def norm_shape(self, w):
        return w.shape[:-2] + (1, 1)

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return w.shape[:-2] + (w.shape[-1], 1)

    def singular_cosine(self, w, v, update):
        return F.cosine_similarity(w @ v, -update.to(v) @ v, dim=-2, eps=eps).squeeze(-1)


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        lmo = torch.sign(g)
        if self.normalized:
            d_out, d_in = g.shape
            lmo /= d_in
        return lmo

    def dual_norm(self, g):
        norm = torch.sum(g.abs())
        if self.normalized:
            d_out, d_in = g.shape
            norm /= d_in
        return norm

    def norm(self, w, v, repeat=1):
        norm = torch.max(w.abs())
        if self.normalized:
            d_out, d_in = w.shape
            norm *= d_in
        return norm, v

    def init(self, w, init_dtype=torch.float64):
        v = w.new_empty((0,))
        return self.norm(w, v)

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return (0,)


norm_dict = {
    'Spectral': Spectral,
    'Sign': Sign,
}


class Scion(torch.optim.Optimizer):
    """Scion optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        weight_decay (float, optional): Weight decay coefficient to be muliplied by the LR. WD * LR
            corresponds to the "learning rate" of the original constrained Scion.
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv',
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)

    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'lr': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'lr': radius*60.0,
        ... }]
        >>> optimizer = Scion(optim_groups, lr=2**-12, momentum=0.1)
    """
    state_keys = ('norm', 'singular', 'momentum_buffer')

    def __init__(self, params, defaults, rank=0, world_size=1):
        if defaults.get('lr', 1e-3) < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if defaults.get('momentum', 1.0) < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if defaults.get('norm_kwargs') is None:
            defaults['norm_kwargs'] = {}
        self.rank = rank
        self.world_size = world_size
        super().__init__(params, defaults)
        self.register_state_dict_pre_hook(self.sync_state)
        self.register_load_state_dict_post_hook(self.remove_unused_keys)

    def assigned_parameters(self):
        """Simple round-robin"""
        index = 0
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                if self.rank == index % self.world_size:
                    yield group, norm_backend, p
                index += 1

    def not_assigned_parameters(self):
        index = 0
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                if self.rank != index % self.world_size:
                    yield group, norm_backend, p
                index += 1

    def remove_unused_keys(self, *_):
        if self.world_size == 1:
            return
        for group, norm_backend, p in self.not_assigned_parameters():
            state = self.state[p]
            for key in self.state_keys:
                state.pop(key, None)

    def sync_state(self, *_):
        for key in self.state_keys:
            self.sync_state_for(key)

    def sync_state_for(self, key):
        if self.world_size == 1:
            return
        index = 0
        buffer = []
        assigned_tensor = None
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            shape_f = getattr(norm_backend, key + '_shape')
            for p in group['params']:
                state = self.state[p]
                if key not in state:
                    state[key] = p.new_empty(shape_f(p.data))
                tensor = state[key]
                buffer.append(tensor)
                if self.rank == index % self.world_size:
                    assigned_tensor = tensor
                if len(buffer) == self.world_size:
                    dist.all_gather(buffer, assigned_tensor)
                    buffer.clear()
                    assigned_tensor = None
                index += 1
        if buffer:
            padding = self.world_size - len(buffer) % self.world_size
            buffer.extend(p.new_empty((0,)) for _ in range(padding))
            if assigned_tensor is None:
                assigned_tensor = buffer[self.rank]
            dist.all_gather(buffer, assigned_tensor)

    def sync_params(self):
        if self.world_size == 1:
            return
        index = 0
        buffer = []
        assigned_p = None
        for group in self.param_groups:
            for p in group['params']:
                buffer.append(p.data)
                if self.rank == index % self.world_size:
                    assigned_p = p.data
                if len(buffer) == self.world_size:
                    dist.all_gather(buffer, assigned_p)
                    buffer.clear()
                    assigned_p = None
                index += 1
        if buffer:
            padding = self.world_size - len(buffer) % self.world_size
            buffer.extend(p.new_empty((0,)) for _ in range(padding))
            if assigned_p is None:
                assigned_p = buffer[self.rank]
            dist.all_gather(buffer, assigned_p)

    @torch.no_grad()
    def step(self):
        for group, norm_backend, p in self.assigned_parameters():
            lr = group['lr']
            momentum = group['momentum']
            wd = lr * group['weight_decay']
            g = p.grad
            if g is None:
                continue
            state = self.state[p]

            if momentum != 1:
                buf = state['momentum_buffer']
                buf.mul_(1-momentum).add_(g, alpha=momentum)
                g = buf

            update = norm_backend.lmo(g)

            state['norm'], state['singular'] = norm_backend.norm(p, state['singular'], repeat=1)
            p.data.mul_(1-wd)
            p.data.add_(update, alpha=-lr)

        self.sync_params()

    def report_norms(self):
        self.sync_state_for('norm')
        spectral = []
        sign = []
        for group in self.param_groups:
            for p in group['params']:
                norm = self.state[p]['norm']
                if group['norm'].startswith('Spectral'):
                    spectral.extend(norm.flatten().tolist())
                else:
                    sign.append(norm.item())
        return math.prod(spectral) ** (1 / len(spectral)), math.fsum(sign) / len(sign)

    def init(self):
        init_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for group, norm_backend, p in self.assigned_parameters():
            init_func = norm_backend.init
            state = self.state[p]
            state['norm'], state['singular'] = init_func(p, init_dtype=init_dtype)
            if group['momentum'] != 1:
                state['momentum_buffer'] = torch.zeros_like(p)


# Polar Express (https://arxiv.org/abs/2505.16932) w/ eps to prevent divide-by-zero
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim = True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(itertools.repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

if not torch.backends.mps.is_available():
    PolarExpress = torch.compile(PolarExpress)
    Spectral.lmo = torch.compile(Spectral.lmo)

def zeroth_power_via_svd(G):
    U, S, V = G.svd()
    return U @ V.T
