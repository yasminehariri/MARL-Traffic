"""Checks the E1 no-communication control (env/comm_model.py E1NoCommModel, train_4x4.py --no-comm).

1. E1 itself is unchanged: with the same weights, the current E1CommModel gives exactly the same
   actions and values as the E1 model of the commit before the control was added (git rev in BASE).
2. The control ignores its neighbours: changing every neighbour observation changes nothing in the
   control's actions or values, while it does change E1's.
3. No information can flow through messages in the control: the gradient of its actions and values
   with respect to the message encoder is exactly zero (in E1 it is not), while the diversity loss
   (on each agent's own message) still reaches the encoder in both, as in E1 training.
4. Same network: both models have exactly the same parameters (names and shapes).
5. The checkpoint flag is read correctly: the published E1 checkpoint is not a control; a run
   folder whose run_info.json says "no_comm": true is.

Run from the repo root:  python tests/test_nocomm.py
"""
import importlib.util, json, os, subprocess, sys, tempfile

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "682f471"     # last commit before the control was added
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from env.comm_model import E1CommModel, E1NoCommModel, OBS_DIM, MAX_NEIGHBORS  # noqa: E402
from mgutil import no_comm_for  # noqa: E402

old_src = subprocess.run(["git", "-C", REPO, "show", f"{BASE}:env/comm_model.py"], capture_output=True,
                         text=True, check=True).stdout
tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "old_comm_model.py"), "w").write(old_src)
spec = importlib.util.spec_from_file_location("old_comm_model", os.path.join(tmp, "old_comm_model.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

OBS = Box(-1, 2, (OBS_DIM * (1 + MAX_NEIGHBORS) + MAX_NEIGHBORS,), np.float32)
ACT = Discrete(4)
make = lambda cls: cls(OBS, ACT, 4, {}, "m")


def batch(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    own = torch.rand(n, OBS_DIM, generator=g)
    nb = torch.rand(n, MAX_NEIGHBORS * OBS_DIM, generator=g)
    k = torch.randint(2, 5, (n,), generator=g)                        # 2, 3 or 4 real neighbours, as in the grid
    mask = (torch.arange(MAX_NEIGHBORS)[None, :] < k[:, None]).float()
    nb = nb * mask.repeat_interleave(OBS_DIM, dim=1)                   # padding slots are zeros, as in the env
    return torch.cat([own, nb, mask], dim=1)


def run(m, x):
    logits, _ = m({"obs": x}, [], None)
    return logits, m.value_function()


torch.manual_seed(1)
e1 = make(E1CommModel)
e1_old = make(old.E1CommModel)
e1_old.load_state_dict(e1.state_dict())
nc = make(E1NoCommModel)
nc.load_state_dict(e1.state_dict())
x = batch()

# 1
for seed in range(3):
    xb = batch(seed=seed)
    a, b = run(e1, xb), run(e1_old, xb)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "E1 changed"
print("1 E1 unchanged: OK (bit-identical actions and values to the E1 model before the control was added)")

# 2
x2 = x.clone()
x2[:, OBS_DIM:OBS_DIM * (1 + MAX_NEIGHBORS)] = torch.rand(x.shape[0], MAX_NEIGHBORS * OBS_DIM) * \
    x[:, -MAX_NEIGHBORS:].repeat_interleave(OBS_DIM, dim=1)
n1, n2 = run(nc, x), run(nc, x2)
assert torch.equal(n1[0], n2[0]) and torch.equal(n1[1], n2[1]), "control reacts to neighbours"
e_a, e_b = run(e1, x), run(e1, x2)
assert not torch.allclose(e_a[0], e_b[0]), "E1 should react to neighbours"
own_only = x.clone()
own_only[:, OBS_DIM:OBS_DIM * (1 + MAX_NEIGHBORS)] = 0
assert torch.equal(run(nc, own_only)[0], n1[0])
print(f"2 control ignores neighbours: OK (all neighbour inputs replaced: control max change 0.0, "
      f"E1 max change {float((e_a[0] - e_b[0]).abs().max().detach()):.3f})")

# 3
for m, want_zero in ((nc, True), (e1, False)):
    m.zero_grad()
    logits, v = run(m, x)
    (logits.sum() + v.sum()).backward()
    g = sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None)
    assert (g == 0.0) == want_zero, (type(m).__name__, g)
for m in (nc, e1):
    m.zero_grad()
    m.encoder(x[:, :OBS_DIM]).pow(2).sum().backward()     # the diversity loss's path: encoder(own obs)
    assert sum(float(p.grad.abs().sum()) for p in m.encoder.parameters()) > 0
print("3 no information through messages: OK (action/value gradient into the encoder is exactly 0 in the "
      "control, non-zero in E1; the diversity-loss path reaches the encoder in both)")

# 4
s1 = {k: tuple(v.shape) for k, v in e1.state_dict().items()}
s2 = {k: tuple(v.shape) for k, v in nc.state_dict().items()}
assert s1 == s2
print(f"4 same network: OK ({sum(np.prod(s) for s in s1.values())} parameters, identical names and shapes)")

# 5
pub = os.path.join(REPO, "checkpoints", "4x4", "E1_local_mg5", "checkpoints", "ckpt_000977")
assert os.path.exists(pub) and no_comm_for(pub) is False
d = tempfile.mkdtemp()
os.makedirs(os.path.join(d, "checkpoints", "ckpt_000001"))
json.dump({"cond": "E1", "no_comm": True}, open(os.path.join(d, "run_info.json"), "w"))
assert no_comm_for(os.path.join(d, "checkpoints", "ckpt_000001")) is True
print("5 checkpoint flag: OK (published E1 = with communication; run_info no_comm=true = control)")
print("ALL NO-COMM CHECKS PASSED")
