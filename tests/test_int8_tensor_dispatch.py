"""Dispatch tests for TensorWise INT8 layout operations."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from comfy_kitchen.tensor import int8 as int8_tensor


def _inputs():
    return (
        torch.randn(2, 4),
        torch.randint(-128, 127, (3, 4), dtype=torch.int8),
        torch.tensor(0.01),
        torch.randn(3),
    )


@pytest.mark.parametrize("missing_compiler", [False, True])
def test_dispatch_int8_linear_eager_uses_registry_when_not_compiling(monkeypatch, missing_compiler):
    """Eager dispatch bypasses the custom op whether torch.compiler exists or not.

    ``missing_compiler=True`` covers PyTorch builds without the module at all,
    where ``is_compiling`` would otherwise raise on access.
    """
    x, weight, weight_scale, bias = _inputs()
    expected = torch.randn(2, 3)
    calls = []

    def implementation(**kwargs):
        calls.append(kwargs)
        return expected

    if missing_compiler:
        monkeypatch.setattr(torch, "compiler", None)
    else:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        int8_tensor.registry,
        "get_implementation",
        lambda name, kwargs: implementation,
    )
    monkeypatch.setattr(
        torch.ops.comfy_kitchen,
        "int8_linear",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("custom op called")),
    )

    actual = int8_tensor._dispatch_int8_linear(
        x, weight, weight_scale, bias, torch.float16, False, 256
    )

    assert actual is expected
    assert calls == [
        {
            "x": x,
            "weight": weight,
            "weight_scale": weight_scale,
            "bias": bias,
            "out_dtype": torch.float16,
            "convrot": False,
            "convrot_groupsize": 256,
            "input_act": None,
        }
    ]


def test_dispatch_int8_linear_eager_resolves_registry_on_every_call(monkeypatch):
    x, weight, weight_scale, bias = _inputs()
    implementations = iter((lambda **kwargs: "first", lambda **kwargs: "second"))
    resolutions = []

    def resolve(name, kwargs):
        resolutions.append((name, kwargs))
        return next(implementations)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(int8_tensor.registry, "get_implementation", resolve)

    args = (x, weight, weight_scale, bias, torch.float32, False, 256)
    assert int8_tensor._dispatch_int8_linear(*args) == "first"
    assert int8_tensor._dispatch_int8_linear(*args) == "second"
    assert [name for name, _ in resolutions] == ["int8_linear", "int8_linear"]


def test_dispatch_int8_linear_compiling_keeps_custom_op(monkeypatch):
    x, weight, weight_scale, bias = _inputs()
    expected = torch.randn(2, 3)
    calls = []

    def custom_op(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        int8_tensor.registry,
        "get_implementation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("registry called")),
    )
    monkeypatch.setattr(torch.ops.comfy_kitchen, "int8_linear", custom_op)

    actual = int8_tensor._dispatch_int8_linear(
        x, weight, weight_scale, bias, torch.bfloat16, True, 128
    )

    assert actual is expected
    assert calls == [(x, weight, weight_scale, bias, 2, True, 128)]


def test_dispatch_int8_linear_fake_tensor_keeps_custom_op(monkeypatch):
    x, weight, weight_scale, bias = _inputs()
    calls = []

    with FakeTensorMode() as mode:
        fake_x = mode.from_tensor(x)

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        monkeypatch.setattr(
            int8_tensor.registry,
            "get_implementation",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("registry called")),
        )
        monkeypatch.setattr(
            torch.ops.comfy_kitchen,
            "int8_linear",
            lambda *args: calls.append(args) or fake_x.new_empty(2, 3),
        )

        actual = int8_tensor._dispatch_int8_linear(
            fake_x, weight, weight_scale, bias, torch.float32, False, 256
        )

    assert tuple(actual.shape) == (2, 3)
    assert calls == [(fake_x, weight, weight_scale, bias, 0, False, 256)]
