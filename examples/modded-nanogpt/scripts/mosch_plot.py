import collections, decimal, math, os, pathlib, statistics, sys, torch, pickle, collections, statistics, io
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.rcParams['text.usetex'] = True

REPO = "$HOME/Downloads/scion/"
GPT_DIR = os.path.join(REPO, "examples/modded-nanogpt/")

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

branch = 'exp-time'

preface = f"""#!/bin/bash

TRAIN={os.path.join(GPT_DIR, "train.py")}
PYTHON="{PYTHON}"

git -C {REPO} checkout {branch}
"""

prefix = ENV + "$PYTHON $TRAIN "

def read_final_loss(p, steps):
    LAST_CKPT = 'state_step%06d.pt' % steps
    ckpt_path = os.path.join(p, LAST_CKPT)
    val_loss = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        val_loss = ckpt['val_loss']
    return val_loss


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
        self.losses = collections.deque()
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
            self.losses.append(loss)

        if done:
            while True:
                nxt, nxt_ok = self.next_value()
                if not nxt_ok:
                    break
                nxt, nxt_cmds, loss = self.test_value(nxt)
                if loss is None:
                    break
                self.values.append(nxt)
                self.losses.append(loss)
            while True:
                prev, prev_ok = self.prev_value()
                if not prev_ok:
                    break
                prev, prev_cmds, loss = self.test_value(prev)
                if loss is None:
                    break
                self.values.appendleft(prev)
                self.losses.appendleft(loss)

        print(self.values, self.losses)

        if done and len(self.values) >= 2:
            pen, ult = self.losses[-2], self.losses[-1]

        if done and (len(self.values) < 2 or pen > ult) and nxt_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {pen} > {ult}:")
                commands.append('')
            done = False
            for nxt_cmd in nxt_cmds: commands.append(nxt_cmd)

        if done and len(self.values) >= 2:
            first, second = self.losses[0], self.losses[1]

        if done and (len(self.values) < 2 or first < second) and prev_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {first} < {second}:")
                commands.append('')
            done = False
            for prev_cmd in prev_cmds: commands.append(prev_cmd)

        if done:
            final_loss, index = min((loss, i) for i, loss in enumerate(self.losses))
            best_val = self.values[index]
            commands.append('')
            commands.append(f"# {best_val=}, {final_loss=}")
            commands.append(f"# {self.curr=}")

        return best_val, commands, final_loss


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
        self.res = {}

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        const_tuner = MoDecayConstAutoTuner(self.diff, self.curr | val, self.f)
        best_const, cmds, val_loss = const_tuner.optimize()
        mdc, losses = const_tuner.values, const_tuner.losses
        if any(losses):
            mdc = [d[self.key] or 0.0 for d in mdc]
            losses = [loss.item() for loss in losses]
            mo = float(val['momentum'])
            end_mo = [mo * math.exp(-x * N_STEP) for x in mdc]
            self.res[mo] = (end_mo, losses)
        val |= best_const
        commands.extend(cmds)
        return val, commands, val_loss  # All commands const_tuner ordered are necessary.

    def set_s_mo(self, val):
        """Set s_mo when necessary to keep it constant throughout tuning"""
        val['s_mo'] = None if almost_eq(self.s_mo, val['momentum']) else self.s_mo

    def next_value(self):
        nxt, ok = super().next_value()
        if ok:
            self.set_s_mo(nxt)
            nxt[self.key] = self.values[-1].get(self.key)
        return nxt, ok

    def prev_value(self):
        prev, ok = super().prev_value()
        if ok:
            self.set_s_mo(prev)
            prev[self.key] = self.values[0].get(self.key)
        return prev, ok

class EndMoRatioAutoTuner(AutoTuner):

    def __init__(self, factor, curr, f):
        self.key = 'q'
        self.factor = factor
        self.max_ratio = 1 / float(curr['momentum'])  # Doesn't make sense to have momentum > 1, right?
        super().__init__(initial_values=[{self.key: curr.get(self.key)}], curr=curr, f=f)

    def next_value(self):
        inv = self.values[-1].get(self.key) or 0.0
        steps = self.curr.get('steps') or N_STEP
        curr_ratio = 1 / (1 + steps * inv)
        prev_ratio = curr_ratio / self.factor
        new_inv = (1/prev_ratio - 1.) / steps
        if almost_eq(new_inv, 0.0):
            new_inv = None
        return {self.key: new_inv}, True

    def prev_value(self):
        inv = self.values[0].get(self.key) or 0.0
        steps = self.curr.get('steps') or N_STEP
        curr_ratio = 1 / (1 + steps * inv)
        if almost_eq(curr_ratio, self.max_ratio):
            return None, False
        next_ratio = min(curr_ratio * self.factor, self.max_ratio)
        new_inv = (1/next_ratio - 1.) / steps
        if almost_eq(new_inv, 0.0):
            new_inv = None
        return {self.key: new_inv}, True


def copy_end_mo(curr, new_mo, key='q'):
    inv = curr.get(key) or 0.0
    steps = curr.get('steps') or N_STEP
    end_mo = float(curr['momentum']) / (1. + steps * inv)
    # new_mo / (1 + steps * inv) = end_mo
    # new_mo / end_mo = 1 + steps * inv
    # inv = (new_mo / end_mo - 1) / steps
    new_inv = (float(new_mo) / end_mo - 1.) / steps
    return None if almost_eq(new_inv, 0.0) else new_inv


class LogTimeMoschAutoTuner(MomentumAutoTuner):
    """Nested AutoTuner for momentum schedule"""
    def __init__(self, factor, curr, f):
        self.key = 'q'
        self.factor = factor
        super().__init__(curr, f)
        self.s_mo = self.curr.get('s_mo')
        if self.s_mo is None:
            self.s_mo = self.curr['momentum']
        self.res = {}

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        ratio_tuner = EndMoRatioAutoTuner(self.factor, self.curr | val, self.f)
        best_ratio, cmds, val_loss = ratio_tuner.optimize()
        q, losses = ratio_tuner.values, ratio_tuner.losses
        if any(losses):
            q = [d[self.key] or 0.0 for d in q]
            losses = [loss.item() for loss in losses]
            mo = float(val['momentum'])
            end_mo = [mo / (1 + x * N_STEP) for x in q]
            self.res[mo] = (end_mo, losses)
        val |= best_ratio
        commands.extend(cmds)
        return val, commands, val_loss  # All commoands ratio_tuner ordered are necessary.

    def set_s_mo(self, val):
        """Set s_mo when necessary to keep it constant throughout tuning"""
        val['s_mo'] = None if almost_eq(self.s_mo, val['momentum']) else self.s_mo

    def next_value(self):
        nxt, ok = super().next_value()
        if ok:
            nxt[self.key] = copy_end_mo(self.values[-1], nxt['momentum'], self.key)
            self.set_s_mo(nxt)
        return nxt, ok

    def prev_value(self):
        prev, ok = super().prev_value()
        if ok:
            prev[self.key] = copy_end_mo(self.values[0], prev['momentum'], self.key)
            self.set_s_mo(prev)
        return prev, ok


# None is tombstone value, '' (empty string) is for store_true flags
# Modified from rerun_mosch.sh
default = {'row_norm': None, 'steps': None, 'corrected': '', 'momentum': 0.02, 'lr': 0.009336277932650495, 'sign_lr': 0.732421875, 'c_sq': 2.899169921875, 'wd': None, 'sign_wd': 0.0003333333333333333, 'nesterov': '', 'cos_power': None, 'power': 1.2, 'mdc': None, 'q': None, 's_mo': None} | {'momentum': decimal.Decimal('0.02'), 'lr': 0.009336277932650495, 'mdc': None}
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figheight(4)
f.set_figwidth(10)

file_like = io.StringIO()
mosch_full = dict(default)
mosch_full['momentum'] = float(mosch_full['momentum'])
diff = math.log(2) / 2 / BUDGET
tuner = MoschAutoTuner(diff=diff, curr=mosch_full, f=file_like)
mosch_full, final_val_loss = tuner.run()
res1 = tuner.res

mosch_full = dict(default)
mosch_full['momentum'] = float(mosch_full['momentum'])
factor = 2 ** (N_STEP / BUDGET / 2)  # Same granularity as the exp-time counterpart
tuner = LogTimeMoschAutoTuner(factor=factor, curr=mosch_full, f=file_like)
mosch_full, final_val_loss = tuner.run()
res2 = tuner.res

def mosch_plot(res, ax, colors):
    table = sorted(res.items())
    colors = [colors[mo] for mo, _ in table]
    x, y = [], []
    for (mo, (end_mo, losses)), color in zip(table, colors):
        ax.plot(end_mo, losses, color=color, label=f"$\\alpha = {mo}$")
        i = min(range(len(losses)), key=lambda i: losses[i])
        x.append(end_mo[i])
        y.append(losses[i])

    ax.scatter(x, y, s=20, edgecolors=colors, facecolors='w', zorder=10, clip_on=False)
    ax.set_xscale('log')
    ax.legend()
    ax.set(xlabel='End momentum $\\alpha$')
    ax.set(ylabel='Val. loss')

mos = sorted(res1 | res2)
cmap = mpl.colormaps['viridis']
colors = dict(zip(mos, cmap(np.linspace(0, 1, len(mos)))))

mosch_plot(res1, ax1, colors)
mosch_plot(res2, ax2, colors)

plt.tight_layout()
plt.savefig('mosch_plot.png')
