import collections, decimal, math, os, pathlib, statistics, sys, torch
from dataclasses import dataclass
from typing import Optional

REPO = "/home/jason-chou/Downloads/scion/"
GPT_DIR = os.path.join(REPO, "examples/modded-nanogpt/")

# For testing:

N_STEP = 21
PYTHON = "torchrun"
folder = "test"
fixed = dict(
    input_bin=f'"{os.path.join(GPT_DIR, f"data/{folder}/fineweb_edu_train_*.bin")}"',  # TODO: Move data out of the repo folder
    input_val_bin=f'"{os.path.join(GPT_DIR, f"data/{folder}/fineweb_edu_val_*.bin")}"', batch_size=1, device_batch_size=1, val_tokens=0, sequence_length=4)

# For production:

# N_STEP = 190734 # Roughly speaking, to be filled in
# BS = 512
# N_THREADS = 208
# PYTHON = f"NUMEXPR_MAX_THREADS={N_THREADS} torchrun --standalone --nproc_per_node=8"
# folder = "fineweb_edu_100BT-shuffled"
# fixed = dict(
#     input_bin=os.path.join(GPT_DIR, f"data/{folder}/fineweb_edu_train_*.bin"),
#     input_val_bin=os.path.join(GPT_DIR, f"data/{folder}/fineweb_edu_val_*.bin"), batch_size=BS, device_batch_size=BS // 8, val_tokens=0)

LAST_CKPT = 'state_step%06d.pt' % N_STEP

def read_final_loss(p):
    ckpt_path = os.path.join(p, LAST_CKPT)
    val_loss = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        val_loss = ckpt['val_loss']
    return val_loss

branch = 'log-time'

preface = f"""#!/bin/bash

TRAIN={os.path.join(GPT_DIR, "train.py")}
PYTHON="{PYTHON}"

git -C {REPO} checkout {branch}
"""

prefix = "$PYTHON $TRAIN "

def run_name(opt, d):
    l = [opt]
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

def flags(d):
    l = []
    for k, v in d.items():
        if v is not None:
            l.append('--' + k.replace('_', '-'))
            if v != '':
                l.append(str(v))
    return ' '.join(l)


def test_params(curr, fixed=fixed, opt='scion', prefix=prefix, path='logs/'):
    name = run_name(opt, curr)
    path_name = os.path.join(path, name)
    command = prefix + flags(curr | fixed | dict(name=name))
    val_loss = read_final_loss(path_name)
    if val_loss is not None:
        command = '# ' + command  # Done
    return command, val_loss


# Due to the naming convention AutoTuner can't distinguish beyond 3 significant digits.
# Should be sufficient given the grid granularity.
def almost_eq(x, y):
    return f"{x:.3g}" == f"{y:.3g}"


class AutoTuner:

    def __init__(self, initial_values, curr, f):
        self.initial_values = initial_values
        self.curr = dict(curr)
        self.f = f

    def next_value(self):
        return None, False

    def prev_value(self):
        return None, False

    def test_value(self, val):
        to_test = self.curr | val
        command, val_loss = test_params(curr=to_test)
        return val, command, val_loss

    def run(self):
        best_val, commands, val_loss = self.optimize()
        for command in commands:
            print(command, file=self.f)
        return self.curr | best_val, val_loss

    def optimize(self):

        done = True
        commands = []
        self.values = collections.deque()
        losses = collections.deque()
        final_loss = None
        best_val = {}

        commands.append('')
        commands.append(f"# {self.initial_values=}")
        commands.append('')

        for val in self.initial_values:
            val, cmd, loss = self.test_value(val)
            done = done and bool(loss)
            self.values.append(val)
            commands.append(cmd)
            losses.append(loss)

        if done:
            while True:
                nxt, nxt_ok = self.next_value()
                if not nxt_ok:
                    break
                nxt, nxt_cmd, loss = self.test_value(nxt)
                if loss is None:
                    break
                self.values.append(nxt)
                losses.append(loss)
            while True:
                prev, prev_ok = self.prev_value()
                if not prev_ok:
                    break
                prev, prev_cmd, loss = self.test_value(prev)
                if loss is None:
                    break
                self.values.appendleft(prev)
                losses.appendleft(loss)

        print(self.values, losses)

        if done and len(self.values) >= 2:
            pen, ult = losses[-2], losses[-1]

        if done and (len(self.values) < 2 or pen > ult) and nxt_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {pen} > {ult}:")
                commands.append('')
            done = False
            commands.append(nxt_cmd)

        if done and len(self.values) >= 2:
            first, second = losses[0], losses[1]

        if done and (len(self.values) < 2 or first < second) and prev_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {first} < {second}:")
                commands.append('')
            done = False
            commands.append(prev_cmd)

        if done:
            final_loss, index = min((loss, i) for i, loss in enumerate(losses))
            best_val = self.values[index]
            commands.append('')
            commands.append(f"# {best_val=}, {final_loss=}")
            commands.append(f"# {self.curr=}")

        return best_val, commands, final_loss


class LRAutoTuner(AutoTuner):

    def __init__(self, key, initial_lr, factor, curr, f):
        self.key = key
        self.factor = factor
        super().__init__(initial_values=[{self.key: initial_lr}], curr=curr, f=f)

    def next_value(self):
        nxt_lr = dict(self.values[-1])
        nxt_lr[self.key] *= self.factor
        return nxt_lr, True

    def prev_value(self):
        prev_lr = dict(self.values[0])
        prev_lr[self.key] /= self.factor
        return prev_lr, True


# class LimitedAutoTuner(LRAutoTuner):

#     def __init__(self, limit, key, initial_lr, factor, curr, f):
#         self.limit = limit
#         super().__init__(key=key, initial_lr=initial_lr, factor=factor, curr=curr, f=f)

#     def next_value(self):
#         if len(self.values) >= self.limit:
#             return None, False
#         return super().next_value()

#     def prev_value(self):
#         if len(self.values) >= self.limit:
#             return None, False
#         return super().prev_value()


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
    return mo


def prev_mo(mo):
    if str(mo)[-1] in '12':
        mo /= 2
    else:
        mo /= 5
        mo *= 2
    return mo


class MomentumAutoTuner(AutoTuner):

    def __init__(self, curr, f):
        self.nesterov = curr.get('nesterov') == ''
        self.lr_eff = curr['lr'] * lr_factor(curr['momentum'], nesterov=self.nesterov)
        init_val = {
            'momentum': decimal.Decimal(str(curr['momentum'])),  # Floating-point precision workaround
            'lr': curr['lr'],
        }
        super().__init__(initial_values=[init_val], curr=curr, f=f)

    def next_value(self):
        mo = self.values[-1]['momentum']
        if mo == 1.0:
            return None, False
        mo = next_mo(mo)
        lr = self.lr_eff / lr_factor(momentum=float(mo), nesterov=self.nesterov)
        return dict(momentum=mo, lr=lr), True

    def prev_value(self):
        mo = self.values[0]['momentum']
        mo = prev_mo(mo)
        lr = self.lr_eff / lr_factor(momentum=float(mo), nesterov=self.nesterov)
        return dict(momentum=mo, lr=lr), True


class EndMoRatioAutoTuner(AutoTuner):

    def __init__(self, factor, curr, f):
        self.key = 'timescale_inv'
        self.factor = factor
        self.max_ratio = 1 / float(curr['momentum'])  # Doesn't make sense to have momentum > 1, right?
        super().__init__(initial_values=[{self.key: curr.get(self.key)}], curr=curr, f=f)

    def next_value(self):
        inv = self.values[-1].get(self.key) or 0.0
        curr_ratio = 1 / (1 + N_STEP * inv)
        prev_ratio = curr_ratio / self.factor
        new_inv = (1/prev_ratio - 1.) / N_STEP
        if almost_eq(new_inv, 0.0):
            new_inv = None
        return {self.key: new_inv}, True

    def prev_value(self):
        inv = self.values[0].get(self.key) or 0.0
        curr_ratio = 1 / (1 + N_STEP * inv)
        if almost_eq(curr_ratio, self.max_ratio):
            return None, False
        next_ratio = min(curr_ratio * self.factor, self.max_ratio)
        new_inv = (1/next_ratio - 1.) / N_STEP
        if almost_eq(new_inv, 0.0):
            new_inv = None
        return {self.key: new_inv}, True


def copy_end_mo(curr, new_mo, key='timescale_inv'):
    inv = curr.get(key) or 0.0
    end_mo = float(curr['momentum']) / (1. + N_STEP * inv)
    # new_mo / (1 + N_STEP * inv) = end_mo
    # new_mo / end_mo = 1 + N_STEP * inv
    # inv = (new_mo / end_mo - 1) / N_STEP
    new_inv = (new_mo / end_mo - 1.) / N_STEP
    return None if almost_eq(new_inv, 0.0) else new_inv


class MoschAutoTuner(MomentumAutoTuner):
    """Nested AutoTuner for momentum schedule"""
    def __init__(self, factor, curr, f):
        self.key = 'timescale_inv'
        self.factor = factor
        super().__init__(curr, f)

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        ratio_tuner = EndMoRatioAutoTuner(self.factor, self.curr | val, self.f)
        best_ratio, cmds, val_loss = ratio_tuner.optimize()
        val |= best_ratio
        commands.extend(cmds)
        return val, commands, val_loss  # All commoands ratio_tuner ordered are necessary.

    def next_value(self):
        nxt, ok = super().next_value()
        if ok:
            nxt[self.key] = copy_end_mo(self.values[-1], nxt['momentum'], self.key)
        return nxt, ok

    def prev_value(self):
        prev, ok = super().prev_value()
        if ok:
            prev[self.key] = copy_end_mo(self.values[0], prev['momentum'], self.key)
        return prev, ok


@dataclass
class Hyperparameters:
    # data hyperparams
    name : Optional[str] = None
    input_bin : str = 'data/fineweb-edu100B/fineweb_edu_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb-edu100B/fineweb_edu_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 0 # number of iterations to run. Defaults to 1 epoch
    seed : Optional[int] = None # change to an int to shuffle files and offsets
    lr : float = 2 ** -12 * 50
    corrected : bool = False
    c_sq : float = 5.79833984375 # (2 - 0.1) / (2 * 0.1) * 2 ** -12 * 50 ** 2
    wd : float = 1 / 50
    sign_lr : float = 2 ** -12 * 3000
    sign_wd : float = 1 / 3000
    grad_clip_norm : float = 1000000. # effectively no clipping
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485761 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons.
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    n_layer : int = 12
    n_head : int = 6 # set as n_embd/128 so head_dim is 128
    n_embd : int = 768
    momentum : float = 0.1
    timescale_inv : float = 0.0
    end_c_sq_mul : float = 1.0
    cautious : bool = False
    cut : bool = False # Use cut_cross_entropy
    nesterov : bool = False
    sign_mo : Optional[float] = None # Momentum for sign paramters, defaults to momentum
    sign_ne : Optional[bool] = None # Nesterov or not for sign parameters, defaults to nesterov


# None is tombstone value, '' (empty string) is for store_true flags
default = dict(
    corrected='',
    momentum=0.1,
    lr=2 ** -12 * 50,
    sign_lr=2 ** -12 * 3000,
    c_sq=5.79833984375,  # (2 - 0.1) / (2 * 0.1) * 2 ** -12 * 50 ** 2
    wd=None,
    sign_wd=1 / 3000,
    nesterov=None,
    timescale_inv=None
)

# old_open = open
# files_opened = []

# def open(file, mode):
#     files_opened.append(file)
#     return old_open(file, mode)

for default['corrected'] in ('', None):

    file_prefix = 'corrected_' if default['corrected'] == '' else ''

    with open(file_prefix + "lr.sh", "w") as f:

        print(preface, file=f)
        print("# LR tuning:", file=f)

        key = 'lr'
        initial_lr = default[key]
        tuner = LRAutoTuner(key, initial_lr, 2 ** 0.5, default, f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    with open(file_prefix + "wd.sh", "w") as f:

        print(preface, file=f)
        print("# Corrected WD tuning:", file=f)

        if default['corrected'] == '':
            key = 'c_sq'
        else:
            key = 'wd'
        initial_wd = default[key]
        tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    with open(file_prefix + "nesterov.sh", "w") as f:

        print(preface, file=f)
        print("# Nesterov or not:", file=f)

        initial_vals = [{k: default.get(k) for k in ['lr', 'nesterov']}]
        nesterov = default.get('nesterov') == ''
        lr_eff = default['lr'] * lr_factor(default['momentum'], nesterov=nesterov)
        new_val = dict(lr=lr_eff / lr_factor(default['momentum'], nesterov=not nesterov), nesterov=None if nesterov else '')
        initial_vals.append(new_val)

        tuner = AutoTuner(initial_values=initial_vals, curr=default, f=f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    with open(file_prefix + "momentum.sh", "w") as f:

        print(preface, file=f)
        print("# Momentum tuning:", file=f)

        tuner = MomentumAutoTuner(default, f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    default['momentum'] = float(default['momentum'])  # Avoid pitfall of inter-op between Decimal & float

    with open(file_prefix + "sign_lr.sh", "w") as f:

        print(preface, file=f)
        print("# Sign LR tuning:", file=f)

        key = 'sign_lr'
        initial_lr = default[key]
        tuner = LRAutoTuner(key, initial_lr, 2 ** 0.5, default, f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    with open(file_prefix + "sign_wd.sh", "w") as f:

        print(preface, file=f)
        print("# Sign WD tuning:", file=f)

        key = 'sign_wd'
        initial_wd = default[key]
        tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
        default, final_val_loss = tuner.run()

    if not final_val_loss:
        sys.exit()

    if default.get('corrected') == '':

        with open(file_prefix + "lr_eff_transfer.sh", "w") as f:

            print(preface, file=f)
            print("# Effective LR transfer:", file=f)

            curr = dict(default)
            mo = curr['momentum']
            lr_eff = curr['lr'] * lr_factor(mo, nesterov=curr.get('nesterov') == '')
            mo = decimal.Decimal(str(mo))  
            mos = collections.deque([mo])
            while len(mos) < 3 and mos[-1] < 1:
                mos.append(next_mo(mos[-1]))
            while len(mos) < 6:
                mos.appendleft(prev_mo(mos[0]))

            factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
            losses = {}
            for curr['nesterov'] in ('', None):
                for curr['momentum'] in mos:
                    for factor in factors:
                        base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                        curr['lr'] = factor * base_lr
                        cmd, val_loss = test_params(curr=curr)
                        losses[curr['nesterov'], curr['momentum'], curr['lr']] = val_loss
                        print(cmd, file=f)

            if not all(losses.values()):
                sys.exit()

        with open(file_prefix + "mo_baseline_comparison.sh", "w") as f:

            print(preface, file=f)
            print("# Double-check after the momentum sweep:", file=f)
            print(losses)

            key = min(losses, key=lambda k: losses[k])
            min_val_loss = losses[key]
            nesterov, momentum, lr = key
            momentum = float(momentum)  # Avoid pitfall of inter-op between Decimal & float
            if default['nesterov'] == nesterov and default['momentum'] == momentum:
                print(file=f)
                print(f"# {(default['nesterov'], default['momentum'], default['lr'])=}, {min_val_loss=}", file=f)
                print(f"# {default=}", file=f)
            else:
                alt = {'nesterov': nesterov, 'momentum': momentum, 'lr': lr}
                tuner = AutoTuner(initial_values=[{}, alt], curr=default, f=f)
                default, final_val_loss = tuner.run()

        if not final_val_loss:
            sys.exit()

    if default.get('corrected') == '':

        corrected_default = dict(default)

        # Prepare uncorrected default
        c_sq = default['c_sq']
        mo, nesterov = default['momentum'], default.get('nesterov') == ''
        initial_wd = lr_factor(mo, nesterov) ** 2 * default['lr'] / c_sq / 2
        # Initial WD guess: half of the initial WD of the best corrected counterpart,
        # so the average throughout the training is about the same
        default['wd'], default['c_sq'] = initial_wd / 2, None

pathlib.Path('done').touch()
print('Done!')

# print(files_opened)
# ['corrected_lr.sh', 'corrected_wd.sh', 'corrected_nesterov.sh', 'corrected_momentum.sh', 'corrected_sign_lr.sh', 'corrected_sign_wd.sh', 'corrected_lr_eff_transfer.sh', 'corrected_mo_baseline_comparison.sh', 'lr.sh', 'wd.sh', 'nesterov.sh', 'momentum.sh', 'sign_lr.sh', 'sign_wd.sh', 'done']
