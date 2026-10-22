"""GILoRaT-style scratch factorization and greedy, function-preserving rank growth.

Adapted from the method in DarvinX/GILoRaT, pinned in references/. The thesis
does not specify generative-model rank settings. This is an explicit ablation.
"""
import torch
from torch import nn


class GrowingLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2, max_rank_divisor=3):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.max_rank = max(1, min(in_features, out_features) // max_rank_divisor)
        self.initial_rank = min(rank,self.max_rank)
        # The reference initializes rank one, then adds zero-initialized rows.
        self.a = nn.Linear(in_features, 1, bias=False)
        self.b = nn.Linear(1, out_features)
        self.register_buffer("ema_abs_grad", torch.zeros_like(self.a.weight))
        self.register_buffer("ema_squared_grad", torch.zeros_like(self.a.weight))
        self._register_tracker()
        self.grow(min(rank, self.max_rank) - 1)

    @property
    def rank(self):
        return self.a.weight.shape[0]

    def _register_tracker(self):
        self.a.weight.register_hook(self._track)

    def _track(self, grad):
        with torch.no_grad():
            self.ema_abs_grad.mul_(.9).add_(grad.detach().abs(), alpha=.1)
            self.ema_squared_grad.mul_(.9).add_(grad.detach().square(), alpha=.1)

    def score(self):
        variance = (self.ema_squared_grad - self.ema_abs_grad.square()).clamp(min=1e-8)
        return (self.ema_abs_grad * variance.sqrt()).mean().item()

    def forward(self, x):
        return self.b(self.a(x))

    @torch.no_grad()
    def grow(self, increment, optimizer=None):
        increment = min(increment, self.max_rank - self.rank)
        if increment <= 0:
            return False
        old_a, old_b = self.a.weight, self.b.weight
        new_a = nn.Parameter(torch.cat([old_a, old_a.new_zeros(increment, self.in_features)], 0))
        new_b = nn.Parameter(torch.cat([old_b, old_b.new_empty(self.out_features, increment).normal_(0, 1)], 1))
        self.a.weight, self.b.weight = new_a, new_b
        self.a.out_features = self.rank
        self.b.in_features = self.rank
        self.ema_abs_grad = torch.cat([self.ema_abs_grad, self.ema_abs_grad.new_zeros(increment, self.in_features)], 0)
        self.ema_squared_grad = torch.cat([self.ema_squared_grad, self.ema_squared_grad.new_zeros(increment, self.in_features)], 0)
        self._register_tracker()
        if optimizer is not None:
            for old, new in [(old_a, new_a), (old_b, new_b)]:
                for group in optimizer.param_groups:
                    group["params"] = [new if p is old else p for p in group["params"]]
                old_state = optimizer.state.pop(old, {})
                migrated = {}
                for key, value in old_state.items():
                    if isinstance(value, torch.Tensor) and value.shape == old.shape:
                        expanded = torch.zeros_like(new)
                        expanded[tuple(slice(0, size) for size in old.shape)] = value
                        migrated[key] = expanded
                    else:
                        migrated[key] = value
                optimizer.state[new] = migrated
        return True


def factorize_transformer(model, settings):
    # Keep text/time/input/output projections dense; isolate transformer factors.
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child.weight.numel() > 2000:
                setattr(parent, name, GrowingLinear(child.in_features, child.out_features, settings.get("initial_rank", 2), settings.get("max_rank_divisor", 3)))
            else:
                replace(child)
    for block in model.blocks:
        replace(block)
    return model


def prepare_for_state(model, state):
    changed = False
    for name, layer in model.named_modules():
        if isinstance(layer, GrowingLinear):
            prefix = f"{name}." if name else ""
            target = state[f"{prefix}a.weight"].shape[0]
            assert layer.initial_rank <= target <= layer.max_rank, "Checkpoint rank is outside configured limits"
            if target < layer.rank:
                # A validation-selected checkpoint may precede the most recent
                # growth. Restore its actual shapes before loading weights and
                # optimizer state; never force it into the final training rank.
                old_a,old_b=layer.a.weight,layer.b.weight
                layer.a.weight=nn.Parameter(old_a.new_empty(target,layer.in_features))
                layer.b.weight=nn.Parameter(old_b.new_empty(layer.out_features,target),requires_grad=old_b.requires_grad)
                layer.a.out_features=target
                layer.b.in_features=target
                layer.ema_abs_grad=layer.ema_abs_grad.new_zeros(target,layer.in_features)
                layer.ema_squared_grad=layer.ema_squared_grad.new_zeros(target,layer.in_features)
                layer._register_tracker()
                layer.a.weight.requires_grad_(old_a.requires_grad)
                changed=True
            else:
                changed = layer.grow(target - layer.rank) or changed
    return changed


def grow_low_score_layers(model, settings, optimizer):
    changed = []
    for name, layer in model.named_modules():
        if isinstance(layer, GrowingLinear) and layer.score() < settings.get("score_threshold", .001):
            if layer.grow(settings.get("increment", 2), optimizer):
                changed.append({"layer": name, "rank": layer.rank})
    return changed


def grow_with_seed(model, settings, optimizer, seed):
    """Isolate new basis draws from the joint encoder's stochastic-depth RNG."""
    devices=sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        cpu=torch.Generator(device="cpu").manual_seed(seed)
        torch.set_rng_state(cpu.get_state())
        for device in devices:
            stream=torch.Generator(device=f"cuda:{device}").manual_seed(seed+device)
            torch.cuda.set_rng_state(stream.get_state(),device)
        return grow_low_score_layers(model,settings,optimizer)


def maximum_rank(model):
    return max((m.rank for m in model.modules() if isinstance(m, GrowingLinear)), default=0)


def orthogonal_penalty(model):
    losses = []
    for layer in model.modules():
        if isinstance(layer, GrowingLinear) and layer.rank > 1:
            gram = layer.a.weight @ layer.a.weight.T
            losses.append(torch.tril(gram, diagonal=-1).abs().sum())
    return sum(losses) if losses else 0.
