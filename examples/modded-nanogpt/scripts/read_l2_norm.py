import collections, decimal, math, os, pathlib, statistics, sys, torch, pickle, collections, statistics
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

OPT = ''
N_STEP = 100835
BUDGET = N_STEP * 3 // 10


def run_name(opt, d):
    l = [opt] if opt else []
    for k, v in d.items():
        if v is not None:
            l.append(k)
            if v != '':
                if type(v) is float:
                    v = f"{v:.3g}"
                else:
                    v = str(v)
                l.append(v)
    return '-'.join(l)


def read_last(curr, opt=OPT, path='logs/'):
    name = run_name(opt, curr)
    path_name = os.path.join(path, name)
    LAST_CKPT = 'state_step%06d.pt' % curr['steps']
    ckpt_path = os.path.join(path_name, LAST_CKPT)
    ckpt = torch.load(ckpt_path, weights_only=False)
    return ckpt


def l2_norms(ckpt):
    hidden = 0.
    for n, p in ckpt['model'].items():
        if n == '_orig_mod.transformer.wte.weight': # weight-tying this is identical to _orig_mod.lm_head.weight
            continue
        elif n == '_orig_mod.lm_head.weight':
            output = torch.linalg.vector_norm(p).item()
        else:
            hidden += torch.sum(p ** 2).item()
    return dict(hidden=math.sqrt(hidden), output=output)


def lr_factor(momentum, nesterov):
    factor = math.sqrt((2 - momentum) / momentum)
    if nesterov:
        factor *= (1 + 4*momentum - 6*momentum**2 + 2*momentum**3) ** -0.5
    return factor


def next_mo(mo):
    if str(mo)[-1] in '15':
        mo *= 2
    else:
        mo *= 5
        mo /= 2
    return mo.normalize()


def prev_mo(mo):
    if str(mo)[-1] in '12':
        mo /= 2
    else:
        mo /= 5
        mo *= 2
    return mo.normalize()


# None is tombstone value, '' (empty string) is for store_true flags
default = {'steps': BUDGET, 'corrected': '', 'momentum': 0.02, 'lr': 0.015125657182366296, 'sign_lr': 0.732421875, 'c_sq': 4.100045423139771, 'wd': None, 'sign_wd': 0.0003333333333333333, 'nesterov': None, 'cos_power': None, 'power': 2.1, 'q': None, 'sign_mo': None} | {'c_sq': 4.100045423139771, 'lr': 0.015125657182366296}
filename = 'gpt_l2_norms.pkl'

mo = default['momentum']
lr_eff = default['lr'] * lr_factor(mo, nesterov=default.get('nesterov') == '')
mo = decimal.Decimal(str(mo))  
mos = collections.deque([mo])
while len(mos) < 3 and mos[-1] < 1:
    mos.append(next_mo(mos[-1]))
while len(mos) < 6:
    mos.appendleft(prev_mo(mos[0]))

def read_l2_norm(mos, default):
    curr = dict(default)
    res = {}
    factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
    for curr['nesterov'] in ('', None):
        for curr['momentum'] in mos:
            for factor in factors:
                base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                curr['lr'] = factor * base_lr
                res[run_name(OPT, curr)] = l2_norms(read_last(curr))

    with open(filename, 'wb') as file:
        pickle.dump(res, file)
    return res

if os.path.exists(filename):
    with open(filename, 'rb') as file:
        res = pickle.load(file)
else:
    res = read_l2_norm(mos, default)


regular = {k: collections.defaultdict(list) for k in ['hidden', 'output']}
nesterov = {k: collections.defaultdict(list) for k in ['hidden', 'output']}

curr = dict(default)
factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
for curr['nesterov'] in ('', None):
    for curr['momentum'] in mos:
        for factor in factors:
            base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
            curr['lr'] = factor * base_lr
            key = run_name(OPT, curr)
            d = regular if curr['nesterov'] is None else nesterov
            for k, v in res[key].items():
                d[k][float(curr['momentum'])].append(v)

fig = plt.figure()
ax = plt.gca()

for label, d, c in [('regular', regular, 'tab:blue'), ('Nesterov', nesterov, 'tab:orange')]:
    for k, v in d.items():
        x = v.keys()
        avg = [statistics.fmean(l) for l in v.values()]
        std = [statistics.stdev(l) for l in v.values()]
        ls = '--' if k == 'output' else '-'
        ax.errorbar(x, avg, yerr=std, linestyle=ls, color=c, label=label)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] for h in handles]
leg1 = ax.legend(handles[::2], labels[::2], bbox_to_anchor=(0.8, 1.0))
ax.add_artist(leg1)
leg2 = ax.legend(handles[:2], ['hidden', 'output'], bbox_to_anchor=(1.0, 1.0))
for line in leg2.legend_handles:
    line.set_color('black')

ax.set_xscale('log')

ax.set(xlabel='Momentum $\\alpha$')
ax.set(ylabel='$L_2$ norm')

plt.tight_layout()
plt.savefig('l2_norm.png')
