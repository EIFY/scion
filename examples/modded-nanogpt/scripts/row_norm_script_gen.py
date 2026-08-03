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
        ckpt = torch.load(ckpt_path, weights_only=True)
        val_loss = ckpt['val_loss']
    return val_loss

branch = 'log-time'

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
    sign_mo=None,
)

# Taken from corrected_mo_baseline_comparison.sh
curr = {'steps': 30250, 'corrected': '', 'momentum': 0.02, 'lr': 0.015125657182366296, 'sign_lr': 0.732421875, 'c_sq': 4.100045423139771, 'wd': None, 'sign_wd': 0.0003333333333333333, 'nesterov': None, 'cos_power': None, 'power': 2.1, 'sign_mo': None}
best_val = {'nesterov': '', 'momentum': 0.02, 'lr': 0.015701685290756325}

default |= curr
default |= best_val

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
