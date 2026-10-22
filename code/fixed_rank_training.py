"""Explicit human-requested rank freeze; weights and native training stay trainable."""
from low_rank import GrowingLinear


def ranks(model):
    return {name: layer.rank for name, layer in model.named_modules()
            if isinstance(layer, GrowingLinear)}


def require_ranks(model, expected):
    actual = ranks(model)
    if actual != expected:
        raise ValueError("Fixed-rank policy mismatch: " + repr(actual))


def frozen_growth(model, settings, optimizer, seed, *, expected):
    # Deliberately do not enter grow_with_seed or alter optimizer/RNG/weights.
    require_ranks(model, expected)
    return []


def install(native, expected):
    """Keep native updates/checkpoints; intercept only authorized growth calls."""
    validation, save = native.validation_b, native.save_checkpoint

    def checked_validation(model, *args, **kwargs):
        require_ranks(model, expected)
        return validation(model, *args, **kwargs)

    def checked_save(path, modules, *args, **kwargs):
        require_ranks(modules["b"], expected)
        return save(path, modules, *args, **kwargs)

    def no_growth(model, settings, optimizer, seed):
        return frozen_growth(model, settings, optimizer, seed, expected=expected)

    native.validation_b = checked_validation
    native.save_checkpoint = checked_save
    native.grow_with_seed = no_growth
