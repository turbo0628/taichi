import numpy as np
import pytest

import taichi as ti
from tests import test_utils


def _test_shared_array_nested_loop():
    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @ti.kernel
    def calc(v: ti.types.ndarray(ndim=1), d: ti.types.ndarray(ndim=1),
             a: ti.types.ndarray(ndim=1)):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @ti.kernel
    def calc_shared_array(v: ti.types.ndarray(ndim=1),
                          d: ti.types.ndarray(ndim=1),
                          a: ti.types.ndarray(ndim=1)):
        ti.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = ti.simt.block.SharedArray((block_dim, ), ti.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                ti.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                ti.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_shared_array_nested_loop():
    arch = ti.lang.impl.get_runtime().prog.config().arch
    if arch == ti.cuda or arch == ti.vulkan:
        _test_shared_array_nested_loop()
    else:
        with pytest.raises(
            ti.TaichiCompilationError,
            match=r"ti\.block\.SharedArray is not supported in current arch .*\. Please use Vulkan or CUDA backends instead\."
        ):
            _test_shared_array_nested_loop()


@test_utils.test(arch=[ti.cuda, ti.vulkan],
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_shared_array_nested_loop_matrix_scalarize():
    _test_shared_array_nested_loop()
