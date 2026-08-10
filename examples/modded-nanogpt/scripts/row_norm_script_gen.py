import collections, decimal, math, os, pathlib, sys, torch

REPO = "$HOME/Downloads/scion/"
GPT_DIR = os.path.join(REPO, "examples/modded-nanogpt/")

# For testing:

# OPT = 'scion'
# BUDGET = 10
# N_STEP = 21
# ENV = ''
# PYTHON = "torchrun"
# folder = "test"
# fixed = dict(
#     input_bin=f'"{folder}/fineweb_edu_train_*.bin"',
#     input_val_bin=f'"{folder}/fineweb_edu_val_*.bin"', batch_size=1, device_batch_size=1, val_tokens=0, sequence_length=4)

# For production:

OPT = ''
N_STEP = 100835
BUDGET = N_STEP * 3 // 10
BS = 1024
N_THREADS = 208
ENV = f"NUMEXPR_MAX_THREADS={N_THREADS} OMP_NUM_THREADS=13 "
PYTHON = "torchrun --standalone --nproc_per_node=8"
folder = "fineweb_edu_100BT-shuffled"
fixed = dict(
    input_bin=f'"{folder}/fineweb_edu_train_*.bin"',
    input_val_bin=f'"{folder}/fineweb_edu_val_*.bin"', batch_size=BS, device_batch_size=BS // 8, val_tokens=0)


def read_final_loss(p, steps):
    LAST_CKPT = 'state_step%06d.pt' % steps
    ckpt_path = os.path.join(p, LAST_CKPT)
    val_loss = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        val_loss = ckpt['val_loss']
    return val_loss

branch = 'exp-time'

preface = f"""#!/bin/bash

TRAIN={os.path.join(GPT_DIR, "train.py")}
PYTHON="{PYTHON}"

git -C {REPO} checkout {branch}
"""

prefix = ENV + "$PYTHON $TRAIN "

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

def flags(d):
    l = []
    for k, v in d.items():
        if v is not None:
            l.append('--' + k.replace('_', '-'))
            if v != '':
                l.append(str(v))
    return ' '.join(l)


def test_params(curr, fixed=fixed, opt=OPT, prefix=prefix, path='logs/'):
    name = run_name(opt, curr)
    path_name = os.path.join(path, name)
    command = prefix + flags(curr | fixed | dict(name=name))
    val_loss = read_final_loss(path_name, steps=curr.get('steps') or N_STEP)
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
        return val, [command], val_loss

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
            val, cmds, loss = self.test_value(val)
            done = done and bool(loss)
            self.values.append(val)
            for cmd in cmds: commands.append(cmd)
            losses.append(loss)

        if done:
            while True:
                nxt, nxt_ok = self.next_value()
                if not nxt_ok:
                    break
                nxt, nxt_cmds, loss = self.test_value(nxt)
                if loss is None:
                    break
                self.values.append(nxt)
                losses.append(loss)
            while True:
                prev, prev_ok = self.prev_value()
                if not prev_ok:
                    break
                prev, prev_cmds, loss = self.test_value(prev)
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
            for nxt_cmd in nxt_cmds: commands.append(nxt_cmd)

        if done and len(self.values) >= 2:
            first, second = losses[0], losses[1]

        if done and (len(self.values) < 2 or first < second) and prev_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {first} < {second}:")
                commands.append('')
            done = False
            for prev_cmd in prev_cmds: commands.append(prev_cmd)

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


class PowerAutoTuner(AutoTuner):

    def __init__(self, key, initial_val, diff, curr, f):
        self.key = key
        self.diff = diff
        super().__init__(initial_values=[{self.key: initial_val}], curr=curr, f=f)

    def next_value(self):
        nxt_val = dict(self.values[-1])
        nxt_val[self.key] += self.diff
        return nxt_val, True

    def prev_value(self):
        prev_val = dict(self.values[0])
        prev_val[self.key] -= self.diff
        if almost_eq(prev_val[self.key], 0.0):
            return None, False
        return prev_val, True


class CosPowerAutoTuner(AutoTuner):

    def __init__(self, key, initial_val, diff, curr, f):
        self.key = key
        self.diff = diff
        if type(initial_val) is float and almost_eq(initial_val, 1.0):
            initial_val = None
        super().__init__(initial_values=[{self.key: initial_val}], curr=curr, f=f)

    def next_value(self):
        nxt_val = self.values[-1][self.key]
        if nxt_val is None:
            nxt_val = 1.0
        nxt_val += self.diff
        if almost_eq(nxt_val, 1.0):
            nxt_val = None
        return {self.key: nxt_val}, True

    def prev_value(self):
        prev_val = self.values[0][self.key]
        if prev_val is None:
            prev_val = 1.0
        prev_val -= self.diff
        if almost_eq(prev_val, 0.0):
            return None, False
        if almost_eq(prev_val, 1.0):
            prev_val = None
        return {self.key: prev_val}, True


class LRPowerAutoTuner(AutoTuner):
    """Nested AutoTuner for LR & schedule power"""
    def __init__(self, factor, initial_value, comp, p_tuner, key, coarse, fine, curr, f):
        self.factor = factor
        self.comp = comp
        self.p_tuner = p_tuner
        self.key = key
        self.coarse = coarse
        self.fine = fine
        super().__init__(initial_values=[initial_value], curr=curr, f=f)

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        tuner = self.p_tuner(key=self.key, initial_val=val[self.key], diff=self.coarse, curr=self.curr | val, f=self.f)
        best_p, cmds, val_loss = tuner.optimize()
        val |= best_p
        commands.extend(cmds)
        if val_loss:
            tuner = self.p_tuner(key=self.key, initial_val=val[self.key], diff=self.fine, curr=self.curr | val, f=self.f)
            best_p, cmds, val_loss = tuner.optimize()
            val |= best_p
            commands.extend(cmds)
        return val, commands, val_loss  # All commands tuner ordered are necessary.

    def next_value(self):
        nxt_lr = dict(self.values[-1])
        nxt_lr['lr'] *= self.factor
        if nxt_lr[self.key] is None:
            nxt_lr[self.key] = 1.0
        nxt_lr[self.key] += self.comp
        return nxt_lr, True

    def prev_value(self):
        prev_lr = dict(self.values[0])
        prev_lr['lr'] /= self.factor
        if prev_lr[self.key] is None:
            prev_lr[self.key] = 1.0
        prev_lr[self.key] -= self.comp
        prev_lr[self.key] = max(prev_lr[self.key], self.coarse)
        return prev_lr, True


class JointCsqLRTuner(AutoTuner):
    """Jointly tune c_sq and lr based on rel. LR"""
    def __init__(self, factor, curr, f):
        assert curr['corrected'] == '', 'Must be a corrected experiment'
        self.factor = factor
        super().__init__(initial_values=[{'c_sq': curr['c_sq'], 'lr': curr['lr']}], curr=curr, f=f)

    def next_value(self):
        curr = self.values[-1]
        next_val = {'c_sq': curr['c_sq'] * self.factor, 'lr': curr['lr'] * math.sqrt(self.factor)}
        return next_val, True

    def prev_value(self):
        curr = self.values[0]
        prev_val = {'c_sq': curr['c_sq'] / self.factor, 'lr': curr['lr'] / math.sqrt(self.factor)}
        return prev_val, True


class MoDecayConstAutoTuner(AutoTuner):

    def __init__(self, diff, curr, f):
        self.key = 'mdc'
        self.diff = diff
        initial_val = curr.get(self.key)
        if type(initial_val) is float and almost_eq(initial_val, 0.0):
            initial_val = None
        super().__init__(initial_values=[{self.key: initial_val}], curr=curr, f=f)

    def next_value(self):
        const = self.values[-1].get(self.key) or 0.0
        const += self.diff
        return {self.key: const}, True

    def prev_value(self):
        const = self.values[0].get(self.key) or 0.0
        if almost_eq(const, 0.0):
            return None, False
        const -= self.diff
        if const < 0.0:
            const = 0.0
        if almost_eq(const, 0.0):
            const = None
        return {self.key: const}, True


class MoschAutoTuner(MomentumAutoTuner):
    """Nested AutoTuner for momentum schedule"""
    def __init__(self, diff, curr, f):
        self.key = 'mdc'
        self.diff = diff
        super().__init__(curr, f)
        self.s_mo = self.curr.get('s_mo')
        if self.s_mo is None:
            self.s_mo = self.curr['momentum']

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        const_tuner = MoDecayConstAutoTuner(self.diff, self.curr | val, self.f)
        best_const, cmds, val_loss = const_tuner.optimize()
        val |= best_const
        commands.extend(cmds)
        return val, commands, val_loss  # All commands const_tuner ordered are necessary.

    def set_s_mo(self, val):
        """Set s_mo when necessary to keep it constant throughout tuning"""
        val['s_mo'] = None if almost_eq(self.s_mo, val['momentum']) else self.s_mo

    def next_value(self):
        nxt, ok = super().next_value()
        if ok: self.set_s_mo(nxt)
        return nxt, ok

    def prev_value(self):
        prev, ok = super().prev_value()
        if ok: self.set_s_mo(prev)
        return prev, ok


# None is tombstone value, '' (empty string) is for store_true flags
default = dict(
    row_norm=None,
    steps=BUDGET,
    corrected='',
    momentum=0.1,
    lr=2 ** -12 * 50,
    sign_lr=2 ** -12 * 3000,
    c_sq=5.79833984375,  # (2 - 0.1) / (2 * 0.1) * 2 ** -12 * 50 ** 2
    wd=None,
    sign_wd=1 / 3000,
    nesterov=None,
    cos_power=None,
    power=None,
    mdc=None,
    s_mo=None,
)

# Modified from corrected_mo_baseline_comparison.sh
curr = {'corrected': '', 'momentum': 0.02, 'lr': 0.015125657182366296, 'sign_lr': 0.732421875, 'c_sq': 4.100045423139771, 'wd': None, 'sign_wd': 0.0003333333333333333, 'nesterov': None, 'cos_power': None, 'power': 2.1}
best_val = {'nesterov': '', 'momentum': 0.02, 'lr': 0.015701685290756325}

default |= curr
default |= best_val

file_prefix = 'rerun_'

with open(file_prefix + "power.sh", "w") as f:

    print(preface, file=f)
    print("# Polynomial decay power tuning:", file=f)

    key = 'power'
    initial_value = {key: default[key], 'lr': default['lr']}
    tuner = LRPowerAutoTuner(
        factor=2**0.5, initial_value=initial_value, comp=0.5, p_tuner=PowerAutoTuner, key=key, coarse=0.2, fine=0.1, curr=default, f=f)
    power_default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

with open(file_prefix + "cos_power.sh", "w") as f:

    print(preface, file=f)
    print("# Cosine decay power tuning:", file=f)

    # According to corrected_cosine_power_comparison.sh the best default gen. cos hyperparameters:
    # Standard power (1.0), 1/sqrt(2) max. lr of that of the best default gen. power hypermeters
    default['power'] = None
    default['lr'] /= 2**0.5

    key = 'cos_power'
    initial_val = 1.0
    initial_value = {key: initial_val, 'lr': default['lr']}
    tuner = LRPowerAutoTuner(
        factor=2**0.5, initial_value=initial_value, comp=0.5, p_tuner=CosPowerAutoTuner, key=key, coarse=0.2, fine=0.1, curr=default, f=f)
    default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

with open(file_prefix + "cosine_power_comparison.sh", "w") as f:

    print(preface, file=f)
    print("# Cosine vs. polynomial decay:", file=f)

    initial_values = [{'cos_power': default['cos_power'], 'lr': default['lr']}]
    initial_values.append({'power': power_default['power'], 'lr': power_default['lr']})
    default['cos_power'] = None

    tuner = AutoTuner(initial_values=initial_values, curr=default, f=f)
    default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

with open("rel_lr.sh", "w") as f:
    print(preface, file=f)
    print("# Near-scale-invariant relative LR:", file=f)
    losses = {}
    curr = dict(default)
    factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
    for lr_f in factors:
        for c_sq_f in factors:
            curr['lr'] = lr_f * math.sqrt(c_sq_f) * default['lr']
            curr['c_sq'] = c_sq_f * default['c_sq']

            cmd, val_loss = test_params(curr=curr)
            losses[curr['lr'], curr['c_sq']] = val_loss
            print(cmd, file=f)

if not all(losses.values()):
    sys.exit()

(default['lr'], default['c_sq']), best_val_loss = min(losses.items(), key=lambda t: t[-1])
print(f"best: {default['lr']=}, {default['c_sq']=}, {best_val_loss=}")

with open(file_prefix + "full.sh", "w") as f:

    print(preface, file=f)
    print("# With all the training tokens:", file=f)
    full = dict(default)
    full['steps'] = None
    cmd, final_val_loss = test_params(curr=full)
    print(cmd, file=f)

if not final_val_loss:
    sys.exit()

with open(file_prefix + "mosch.sh", "w") as f:

    print(preface, file=f)
    print("# Momentum scheduling!", file=f)

    steps = curr.get('steps') or N_STEP
    # 1/factor = exp(-const * steps)
    # factor = exp(const * steps)
    # log(factor) = const * steps
    # const = log(factor) / steps
    diff = math.log(2) / 2 / steps
    tuner = MoschAutoTuner(diff=diff, curr=default, f=f)
    mosch_default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

with open(file_prefix + "mosch_full.sh", "w") as f:

    print(preface, file=f)
    print("# Does the optimal decay const stay the same?", file=f)
    mosch_full = dict(mosch_default)
    mosch_full['steps'] = None
    tuner = MoDecayConstAutoTuner(diff=diff, curr=mosch_full, f=f)
    mosch_full, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

branch = 'row-norm'

preface = f"""#!/bin/bash

TRAIN={os.path.join(GPT_DIR, "train.py")}
PYTHON="{PYTHON}"

git -C {REPO} checkout {branch}
"""

default['row_norm'] = ''
file_prefix = 'row_norm_'

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

    key = 'c_sq'
    initial_wd = default[key]
    tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
    default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

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

with open(file_prefix + "c_sq_lr.sh", "w") as f:

    print(preface, file=f)
    print("# Joint c_sq and lr tuning:", file=f)
    tuner = JointCsqLRTuner(factor=2 ** 0.5, curr=default, f=f)
    default, final_val_loss = tuner.run()

if not final_val_loss:
    sys.exit()

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

pathlib.Path(file_prefix + 'done').touch()
print('Done!')

# print(files_opened)
# ['row_norm_lr.sh', 'row_norm_wd.sh', 'row_norm_sign_lr.sh', 'row_norm_sign_wd.sh', 'row_norm_c_sq_lr.sh', 'row_norm_lr_eff_transfer.sh', 'row_norm_done']
