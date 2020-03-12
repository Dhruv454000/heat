import unittest
import os
import heat as ht
import torch
import numpy as np

from heat.core.tests.test_suites.basic_test import BasicTest

if os.environ.get("DEVICE") == "gpu" and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and ht.torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    ht.torch.cuda.set_device(device)


class TestMemory(BasicTest):
    def test_copy(self):
        tensor = ht.ones(5, device=ht_device)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._DNDarray__array, copied._DNDarray__array)
        self.assertTrue((tensor == copied)._DNDarray__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy("hello world")

    def test_sanitize_memory_layout(self):
        # non distributed, 2D
        a_torch = torch.arange(12, device=device).reshape(4, 3)
        a_heat_C = ht.array(a_torch, device=ht_device)
        a_heat_F = ht.array(a_torch, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_C, "C")
        self.assertTrue_memory_layout(a_heat_F, "F")
        # non distributed, 5D
        a_torch_5d = torch.arange(4 * 3 * 5 * 2 * 1, device=device).reshape(4, 3, 1, 2, 5)
        a_heat_5d_C = ht.array(a_torch_5d, device=ht_device)
        a_heat_5d_F = ht.array(a_torch_5d, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_5d_C, "C")
        self.assertTrue_memory_layout(a_heat_5d_F, "F")
        a_heat_5d_F_sum = a_heat_5d_F.sum(-2)
        a_torch_5d_sum = a_torch_5d.sum(-2)
        self.assert_array_equal(a_heat_5d_F_sum, a_torch_5d_sum)
        # distributed, split, 2D
        size = ht.communication.MPI_WORLD.size
        a_torch_2d = torch.arange(4 * size * 3 * size, device=device).reshape(4 * size, 3 * size)
        a_heat_2d_C_split = ht.array(a_torch_2d, split=0, device=ht_device)
        a_heat_2d_F_split = ht.array(a_torch_2d, split=1, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_2d_C_split, "C")
        self.assertTrue_memory_layout(a_heat_2d_F_split, "F")
        a_heat_2d_F_split_sum = a_heat_2d_F_split.sum(1)
        a_torch_2d_sum = a_torch_2d.sum(1)
        self.assert_array_equal(a_heat_2d_F_split_sum, a_torch_2d_sum)
        # distributed, split, 5D
        a_torch_5d = torch.arange(4 * 3 * 5 * 2 * size * 7, device=device).reshape(
            4, 3, 7, 2 * size, 5
        )
        a_heat_5d_C_split = ht.array(a_torch_5d, split=-2, device=ht_device)
        a_heat_5d_F_split = ht.array(a_torch_5d, split=-2, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_5d_C_split, "C")
        self.assertTrue_memory_layout(a_heat_5d_F_split, "F")
        a_heat_5d_F_split_sum = a_heat_5d_F_split.sum(-2)
        a_torch_5d_sum = a_torch_5d.sum(-2)
        self.assert_array_equal(a_heat_5d_F_split_sum, a_torch_5d_sum)
        # distributed, is_split, 2D
        a_heat_2d_C_issplit = ht.array(a_torch_2d, is_split=0, device=ht_device)
        a_heat_2d_F_issplit = ht.array(a_torch_2d, is_split=1, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_2d_C_issplit, "C")
        self.assertTrue_memory_layout(a_heat_2d_F_issplit, "F")
        a_heat_2d_F_issplit_sum = a_heat_2d_F_issplit.sum(1)
        a_torch_2d_sum = a_torch_2d.sum(1) * size
        self.assert_array_equal(a_heat_2d_F_issplit_sum, a_torch_2d_sum)
        # distributed, is_split, 5D
        a_heat_5d_C_issplit = ht.array(a_torch_5d, is_split=-2, device=ht_device)
        a_heat_5d_F_issplit = ht.array(a_torch_5d, is_split=-2, order="F", device=ht_device)
        self.assertTrue_memory_layout(a_heat_5d_C_issplit, "C")
        self.assertTrue_memory_layout(a_heat_5d_F_issplit, "F")
        a_heat_5d_F_issplit_sum = a_heat_5d_F_issplit.sum(-2)
        a_torch_5d_sum = a_torch_5d.sum(-2) * size
        self.assert_array_equal(a_heat_5d_F_issplit_sum, a_torch_5d_sum)
        # test exceptions
        with self.assertRaises(NotImplementedError):
            ht.zeros_like(a_heat_5d_C_split, order="K")
