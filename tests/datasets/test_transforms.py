import torch
import pytest

import oodd.datasets.transforms as transforms


def test_scale_zero_one():
    t = transforms.Scale()

    r = torch.rand(10, 10) * 10 - 3

    o = t(r)
    
    assert o.min() == 0
    assert o.max() == 1


def test_scale_a_b():
    t = transforms.Scale(a=-1, b=1)

    r = torch.rand(10, 10) * 10 - 3

    o = t(r)
    
    assert o.min() == -1
    assert o.max() == 1


def test_scale_min_val_max_val():
    t = transforms.Scale(min_val=-3, max_val=7)

    r = torch.rand(10, 10) * 10 - 3

    o = t(r)

    assert 0.0 < o.min() < 0.1
    assert 0.9 < o.max() < 1.0


def test_scale_failure():
    with pytest.raises(AssertionError):
        transforms.Scale(a=1)
        transforms.Scale(b=1)
        transforms.Scale(a=1, min_val=1)
        transforms.Scale(b=1, max_val=1)


def test_binarize_resample():
    torch.manual_seed(42)
    t = transforms.Binarize(resample=True)
    
    r = torch.rand(10, 10)
    
    o_1 = t(r)
    o_2 = t(r)
    
    assert o_1.min() == 0
    assert o_1.max() == 1
    assert o_1.unique().numel() == 2

    assert o_2.min() == 0
    assert o_2.max() == 1
    assert o_2.unique().numel() == 2

    assert (o_1 != o_2).any()


def test_binarize_threshold():
    t = transforms.Binarize(threshold=0.5)

    r = torch.rand(10, 10)

    o = t(r)
    o_new = t(r)

    assert o.min() == 0
    assert o.max() == 1
    assert o.unique().numel() == 2
    assert (o == o_new).all()


def test_binarize_failure():
    with pytest.raises(AssertionError):
        t = transforms.Binarize()
        t = transforms.Binarize(resample=True, threshold=0.5)


def test_dequantize():
    torch.manual_seed(42)
    t = transforms.Dequantize()

    r = torch.randint(low=0, high=256, size=(1000,), dtype=torch.float32)

    o_1 = t(r)
    o_2 = t(r)
    
    assert 0.0 < o_1.min() < 1
    assert 255 < o_1.max() < 256
    assert o_1.unique().numel() == 1000  # Should almost always be true

    assert 0.0 < o_2.min() < 1
    assert 255 < o_2.max() < 256
    assert o_2.unique().numel() == 1000  # Should almost always be true

    assert (o_1 != o_2).all()
