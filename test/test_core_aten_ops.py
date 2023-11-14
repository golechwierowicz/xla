import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch
import torch._export

import tempfile
import unittest


class AtenOpTest(unittest.TestCase):

  def test_aten_abs_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.abs(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.abs, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_abs_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.abs(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.abs, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_abs_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.abs(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.abs, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_abs_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.abs(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.abs, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acos_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.acos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acos_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.acos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acos_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.acos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_acos_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.acos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acosh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.acosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acosh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.acosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_acosh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.acosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_acosh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.acosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.acosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__adaptive_avg_pool2d_0(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float32),
        [
            5,
            5,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten._adaptive_avg_pool2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._adaptive_avg_pool2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__adaptive_avg_pool3d_0(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float32),
        [
            5,
            5,
            5,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten._adaptive_avg_pool3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._adaptive_avg_pool3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__adaptive_avg_pool3d_1(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float16),
        [
            5,
            5,
            5,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten._adaptive_avg_pool3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._adaptive_avg_pool3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_add_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.add.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.add.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_add_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.add.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.add.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_add_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.add.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.add.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_add_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.add.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.add.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_addmm_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.addmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.addmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_addmm_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.addmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.addmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_addmm_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.addmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.addmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_addmm_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.addmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.addmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_alias_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.alias(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.alias, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_alias_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.alias(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.alias, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_alias_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.alias(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.alias, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_alias_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.alias(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.alias, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amax_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amax_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amin_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amin_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amin_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_amin_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.amin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.amin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.any(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.any(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.any(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.any(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.any.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.any.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dim_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.any.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dims_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.any.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dims_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.any.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_any_dims_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.any.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.any.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmax_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.argmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmax_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.argmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmax_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.argmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmax_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.argmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.argmin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmin_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.argmin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmin_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.argmin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_argmin_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.argmin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.argmin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_as_strided_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.as_strided(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.as_strided, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_as_strided_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.as_strided(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.as_strided, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_as_strided_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.as_strided(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.as_strided, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_as_strided_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.as_strided(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.as_strided, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_asin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.asin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_asin_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.asin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_asin_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.asin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_asin_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.asin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_asinh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.asinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_asinh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.asinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_asinh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.asinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_asinh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.asinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.asinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_atan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.atan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_atan_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.atan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_atan_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.atan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_atan_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.atan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_atan2_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.atan2(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan2, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_atan2_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.atan2(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atan2, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_atanh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.atanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_atanh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.atanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_atanh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.atanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_atanh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.atanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.atanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_avg_pool2d_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.avg_pool2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.avg_pool2d, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_avg_pool2d_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.avg_pool2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.avg_pool2d, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_avg_pool3d_0(self):
    args = (
        torch.randn((1, 3, 10, 10, 10)).to(torch.float32),
        [
            2,
            2,
            2,
        ],
        [
            2,
            2,
            2,
        ],
        [
            0,
            0,
            0,
        ],
        False,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.avg_pool3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.avg_pool3d, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bitwise_and_Tensor_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.bitwise_and.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bitwise_and.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bitwise_not_0(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.bitwise_not(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bitwise_not, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bitwise_or_Tensor_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.bitwise_or.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bitwise_or.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bitwise_xor_Tensor_0(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.bitwise_xor.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bitwise_xor.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bmm_0(self):
    args = (
        torch.randn((10, 10, 10)).to(torch.float32),
        torch.randn((10, 10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.bmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bmm_1(self):
    args = (
        torch.randn((10, 10, 10)).to(torch.float32),
        torch.randn((10, 10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.bmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bmm_2(self):
    args = (
        torch.randn((10, 10, 10)).to(torch.float16),
        torch.randn((10, 10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.bmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_bmm_3(self):
    args = (
        torch.randint(0, 10, (10, 10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.bmm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.bmm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cat_0(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cat(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cat, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cat_1(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cat(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cat, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cat_2(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cat(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cat, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cat_3(self):
    args = (
        [
            torch.randn((10, 10)).to(torch.float32),
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cat(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cat, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_rand_0(self):
    args = ([
        10,
        10,
    ],)
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.rand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_rand_1(self):
    args = ([
        10,
        10,
    ],)
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.rand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_rand_2(self):
    args = ([
        10,
        10,
    ],)
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.rand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_rand_3(self):
    args = ([
        2,
        1,
    ],)
    kwargs = dict()
    res = torch.ops.aten.rand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__cdist_forward_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        1.0,
        None,
    )
    kwargs = dict()
    res = torch.ops.aten._cdist_forward(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._cdist_forward, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ceil_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.ceil(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ceil, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ceil_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.ceil(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ceil, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ceil_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.ceil(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ceil, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.clamp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.clamp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.clamp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.clamp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((1,)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.clamp.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((1,)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.clamp.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clamp_Tensor_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (1,)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.clamp.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clamp.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clone_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.clone(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clone, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clone_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.clone(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clone, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clone_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.clone(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clone, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_clone_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.clone(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.clone, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_constant_pad_nd_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.constant_pad_nd(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.constant_pad_nd, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_constant_pad_nd_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.constant_pad_nd(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.constant_pad_nd, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_constant_pad_nd_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.constant_pad_nd(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.constant_pad_nd, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_constant_pad_nd_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.constant_pad_nd(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.constant_pad_nd, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_convolution_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        torch.randn((2, 2, 2)).to(torch.float32),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.convolution(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.convolution, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_convolution_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        torch.randn((2, 2, 2)).to(torch.float32),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.convolution(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.convolution, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_convolution_2(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float16),
        torch.randn((2, 2, 2)).to(torch.float16),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.convolution(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.convolution, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_convolution_3(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        torch.randint(0, 10, (2, 2, 2)).to(torch.int32),
        None,
        [
            2,
        ],
        [
            0,
        ],
        [
            1,
        ],
        False,
        [
            0,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.convolution(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.convolution, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cos_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.cos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cos_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.cos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cos_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.cos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_cos_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.cos(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cos, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cosh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.cosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cosh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.cosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cosh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.cosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_cosh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.cosh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cosh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cumsum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cumsum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cumsum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_cumsum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cumsum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cumsum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_cumsum_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cumsum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cumsum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_cumsum_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.cumsum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.cumsum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_diagonal_0(self):
    args = (torch.randn((10, 20)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.diagonal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.diagonal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_diagonal_1(self):
    args = (torch.randn((10, 20)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.diagonal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.diagonal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_diagonal_2(self):
    args = (torch.randint(0, 10, (10, 20)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.diagonal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.diagonal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Scalar_mode_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        0.123,
    )
    kwargs = dict((
        "rounding_mode",
        "trunc",
    ))
    res = torch.ops.aten.div.Scalar_mode(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Scalar_mode, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Scalar_mode_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        0.123,
    )
    kwargs = dict((
        "rounding_mode",
        "trunc",
    ))
    res = torch.ops.aten.div.Scalar_mode(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Scalar_mode, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Scalar_mode_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        0.123,
    )
    kwargs = dict((
        "rounding_mode",
        "trunc",
    ))
    res = torch.ops.aten.div.Scalar_mode(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Scalar_mode, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_div_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.div.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_div_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.div.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_div_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.div.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.div.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Tensor_mode_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict((
        "rounding_mode",
        "trunc",
    ))
    res = torch.ops.aten.div.Tensor_mode(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor_mode, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_div_Tensor_mode_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict((
        "rounding_mode",
        "trunc",
    ))
    res = torch.ops.aten.div.Tensor_mode(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.div.Tensor_mode, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_embedding_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.embedding(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.embedding, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_embedding_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.embedding(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.embedding, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_embedding_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.embedding(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.embedding, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_embedding_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.embedding(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.embedding, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_empty_strided_0(self):
    args = (
        [
            10,
            10,
        ],
        [
            10,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.empty_strided(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.empty_strided, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_eq_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.eq.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.eq.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_eq_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.eq.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.eq.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_eq_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.eq.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.eq.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_eq_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.eq.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.eq.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_erf_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.erf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.erf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_erf_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.erf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.erf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_erf_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.erf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.erf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_erf_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.erf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.erf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_exp_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.exp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.exp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_exp_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.exp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.exp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_exp_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.exp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.exp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_exp_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.exp(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.exp, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_expand_0(self):
    args = (
        torch.randn((10, 1)).to(torch.float32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.expand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_expand_1(self):
    args = (
        torch.randn((10, 1)).to(torch.float32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.expand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_expand_2(self):
    args = (
        torch.randn((10, 1)).to(torch.float16),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.expand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_expand_3(self):
    args = (
        torch.randint(0, 10, (10, 1)).to(torch.int32),
        [
            10,
            10,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.expand(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expand, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_expm1_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.expm1(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expm1, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_expm1_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.expm1(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expm1, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_expm1_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.expm1(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.expm1, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_flip_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.flip(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.flip, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_flip_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.flip(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.flip, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_flip_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.flip(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.flip, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_flip_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.flip(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.flip, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_floor_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.floor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.floor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_floor_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.floor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.floor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_floor_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.floor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.floor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_floor_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.floor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.floor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_fmod_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.fmod.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.fmod.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_fmod_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.fmod.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.fmod.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_fmod_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.fmod.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.fmod.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_full_0(self):
    args = (
        [
            10,
            10,
        ],
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.full(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.full, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gather_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.gather(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gather, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gather_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.gather(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gather, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gather_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.gather(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gather, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gather_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.gather(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gather, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ge_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.ge.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ge.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ge_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.ge.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ge.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ge_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.ge.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ge.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ge_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.ge.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ge.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gelu_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.gelu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gelu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gelu_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.gelu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gelu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_gelu_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.gelu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gelu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_grid_sampler_2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 2, 2, 2)).to(torch.float32),
        0,
        0,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.grid_sampler_2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.grid_sampler_2d, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_grid_sampler_2d_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 2, 2, 2)).to(torch.float32),
        0,
        0,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.grid_sampler_2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.grid_sampler_2d, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gt_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.gt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gt_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.gt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gt_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.gt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_gt_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.gt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.gt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_hardtanh_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.hardtanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.hardtanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_hardtanh_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.hardtanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.hardtanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_hardtanh_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.hardtanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.hardtanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_hardtanh_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.hardtanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.hardtanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_put_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randn((10,)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.index_put(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_put, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_put_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randn((10,)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.index_put(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_put, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_put_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randn((10,)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.index_put(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_put, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_put_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            torch.randint(0, 10, (1,)).to(torch.int64),
        ],
        torch.randint(0, 10, (10,)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.index_put(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_put, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_0(self):
    args = (
        0,
        10,
        [
            1,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_1(self):
    args = (
        0,
        10,
        [
            1,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_2(self):
    args = (
        0,
        10,
        [
            1,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_3(self):
    args = (
        0,
        2,
        [
            2,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_4(self):
    args = (
        0,
        2,
        [
            2,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randint_low_5(self):
    args = (
        0,
        2,
        [
            2,
        ],
    )
    kwargs = dict((
        "device",
        cpu,
    ), (
        "pin_memory",
        False,
    ))
    res = torch.ops.aten.randint.low(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randint.low, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_select_0(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.index_select(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_select, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_select_1(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.index_select(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_select, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_select_2(self):
    args = (
        torch.randn((2, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.index_select(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_select, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_index_select_3(self):
    args = (
        torch.randint(0, 10, (2, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2,)).to(torch.int64),
    )
    kwargs = dict()
    res = torch.ops.aten.index_select(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index_select, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_index_Tensor_0(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        [
            torch.randint(0, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.index.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_index_Tensor_1(self):
    args = (
        torch.randn((2, 10)).to(torch.float32),
        [
            torch.randint(0, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.index.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_index_Tensor_2(self):
    args = (
        torch.randn((2, 10)).to(torch.float16),
        [
            torch.randint(0, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.index.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_index_Tensor_3(self):
    args = (
        torch.randint(0, 10, (2, 10)).to(torch.int32),
        [
            torch.randint(0, 10, (2,)).to(torch.int64),
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.index.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.index.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isinf_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.isinf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isinf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isinf_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.isinf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isinf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isinf_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.isinf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isinf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_isinf_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.isinf(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isinf, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isnan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.isnan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isnan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isnan_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.isnan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isnan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_isnan_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.isnan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isnan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_isnan_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.isnan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.isnan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_le_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.le.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.le.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_le_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.le.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.le.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_le_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.le.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.le.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_le_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.le.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.le.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_leaky_relu_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.leaky_relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.leaky_relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_leaky_relu_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.leaky_relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.leaky_relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_leaky_relu_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.leaky_relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.leaky_relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.log(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.log(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.log(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_log_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.log(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log10_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.log10(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log10, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log10_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.log10(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log10, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_log10_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.log10(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log10, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log1p_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.log1p(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log1p, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log1p_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.log1p(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log1p, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_log1p_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.log1p(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log1p, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log2_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.log2(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log2, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_log2_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.log2(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log2, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_log2_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.log2(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.log2, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__log_softmax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten._log_softmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._log_softmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten__log_softmax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten._log_softmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._log_softmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_and_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_and(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_and, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_and_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_and(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_and, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_and_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_and(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_and, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_and_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_and(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_and, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_not_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.logical_not(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_not, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_not_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.logical_not(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_not, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_not_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.logical_not(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_not, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_not_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.logical_not(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_not, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_or_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_or(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_or, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_or_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_or(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_or, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_or_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_or(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_or, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_or_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_or(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_or, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_xor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_xor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_xor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_xor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_xor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_xor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_xor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_xor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_xor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_logical_xor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.logical_xor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.logical_xor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_lt_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.lt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.lt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_lt_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.lt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.lt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_lt_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.lt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.lt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_lt_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.lt.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.lt.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_max_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.max.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_max_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.max.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_max_dim_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.max.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_max_dim_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.max.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool2d_with_indices_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool2d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool2d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool2d_with_indices_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool2d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool2d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool2d_with_indices_2(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float16),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool2d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool2d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool2d_with_indices_3(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            2,
            2,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool2d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool2d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool3d_with_indices_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool3d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool3d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool3d_with_indices_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool3d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool3d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool3d_with_indices_2(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool3d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool3d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_max_pool3d_with_indices_3(self):
    args = (
        torch.randint(0, 10, (1, 3, 2, 10)).to(torch.int32),
        [
            2,
            2,
            2,
        ],
        [
            1,
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.max_pool3d_with_indices(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.max_pool3d_with_indices, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_maximum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.maximum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.maximum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_maximum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.maximum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.maximum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_maximum_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.maximum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.maximum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_maximum_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.maximum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.maximum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mean_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.mean(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mean, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mean_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.mean(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mean, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mean_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.mean.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mean.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mean_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.mean.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mean.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mean_dim_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.mean.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mean.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_min_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.min.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.min.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_min_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.min.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.min.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_min_dim_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.min.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.min.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_min_dim_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.min.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.min.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_minimum_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.minimum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.minimum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_minimum_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.minimum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.minimum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_minimum_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.minimum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.minimum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_minimum_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.minimum(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.minimum, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mm_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.mm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mm_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.mm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mm_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.mm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mm_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.mm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mm, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mul_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.mul.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mul.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mul_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.mul.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mul.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mul_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.mul.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mul.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_mul_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.mul.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.mul.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__native_batch_norm_legit_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
        None,
        torch.randn((10,)).to(torch.float32),
        torch.randn((10,)).to(torch.float32),
        False,
        1.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._native_batch_norm_legit,
                                   args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__native_batch_norm_legit_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
        None,
        torch.randn((10,)).to(torch.float16),
        torch.randn((10,)).to(torch.float16),
        False,
        1.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._native_batch_norm_legit,
                                   args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__native_batch_norm_legit_no_stats_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        True,
        0.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit.no_stats(*args, **kwargs)
    exported = torch.export.export(
        torch.ops.aten._native_batch_norm_legit.no_stats, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__native_batch_norm_legit_no_stats_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        True,
        0.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit.no_stats(*args, **kwargs)
    exported = torch.export.export(
        torch.ops.aten._native_batch_norm_legit.no_stats, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__native_batch_norm_legit_no_stats_2(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        True,
        0.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit.no_stats(*args, **kwargs)
    exported = torch.export.export(
        torch.ops.aten._native_batch_norm_legit.no_stats, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten__native_batch_norm_legit_no_training_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
        None,
        torch.randn((10,)).to(torch.float32),
        torch.randn((10,)).to(torch.float32),
        1.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit_no_training(*args, **kwargs)
    exported = torch.export.export(
        torch.ops.aten._native_batch_norm_legit_no_training, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten__native_batch_norm_legit_no_training_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
        None,
        torch.randn((10,)).to(torch.float16),
        torch.randn((10,)).to(torch.float16),
        1.0,
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._native_batch_norm_legit_no_training(*args, **kwargs)
    exported = torch.export.export(
        torch.ops.aten._native_batch_norm_legit_no_training, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_dropout_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.0,
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.native_dropout(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_dropout, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_dropout_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.0,
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.native_dropout(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_dropout, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_dropout_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1.0,
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.native_dropout(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_dropout, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_group_norm_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        None,
        None,
        1,
        3,
        20,
        1,
        0.0,
    )
    kwargs = dict()
    res = torch.ops.aten.native_group_norm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_group_norm, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_group_norm_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        None,
        None,
        1,
        3,
        20,
        1,
        0.0,
    )
    kwargs = dict()
    res = torch.ops.aten.native_group_norm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_group_norm, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_native_group_norm_2(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float16),
        None,
        None,
        1,
        3,
        20,
        1,
        0.0,
    )
    kwargs = dict()
    res = torch.ops.aten.native_group_norm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_group_norm, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_native_layer_norm_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            3,
            2,
            10,
        ],
        None,
        None,
        0.0,
    )
    kwargs = dict()
    res = torch.ops.aten.native_layer_norm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_layer_norm, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_native_layer_norm_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            3,
            2,
            10,
        ],
        None,
        None,
        0.0,
    )
    kwargs = dict()
    res = torch.ops.aten.native_layer_norm(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.native_layer_norm, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_ne_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.ne.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ne.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ne_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.ne.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ne.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ne_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.ne.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ne.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_ne_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.ne.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.ne.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_neg_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.neg(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.neg, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_neg_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.neg(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.neg, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_neg_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.neg(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.neg, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_neg_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.neg(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.neg, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten__pdist_forward_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1.0,
    )
    kwargs = dict()
    res = torch.ops.aten._pdist_forward(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._pdist_forward, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_permute_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.permute(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.permute, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_permute_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.permute(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.permute, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_permute_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.permute(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.permute, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_permute_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.permute(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.permute, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_pixel_shuffle_0(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.pixel_shuffle(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pixel_shuffle, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_pixel_shuffle_1(self):
    args = (
        torch.randn((1, 3, 10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.pixel_shuffle(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pixel_shuffle, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_pixel_shuffle_2(self):
    args = (
        torch.randint(0, 10, (1, 3, 10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.pixel_shuffle(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pixel_shuffle, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_pow_Tensor_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.pow.Tensor_Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pow.Tensor_Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_pow_Tensor_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.pow.Tensor_Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pow.Tensor_Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_pow_Tensor_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.pow.Tensor_Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pow.Tensor_Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_pow_Tensor_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.pow.Tensor_Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.pow.Tensor_Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_prod_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.prod(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.prod, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_prod_1(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.prod(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.prod, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_prod_dim_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.prod.dim_int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.prod.dim_int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_prod_dim_int_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.prod.dim_int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.prod.dim_int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_randn_0(self):
    args = ([
        2,
        1,
    ],)
    kwargs = dict()
    res = torch.ops.aten.randn(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.randn, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reciprocal_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.reciprocal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reciprocal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reciprocal_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.reciprocal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reciprocal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reciprocal_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.reciprocal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reciprocal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_reciprocal_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.reciprocal(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reciprocal, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_reflection_pad1d_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad1d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad1d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_reflection_pad1d_1(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad1d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad1d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reflection_pad2d_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reflection_pad2d_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_reflection_pad2d_2(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_reflection_pad3d_0(self):
    args = (
        torch.randn((3, 3, 3, 3, 3, 3)).to(torch.float32),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_reflection_pad3d_1(self):
    args = (
        torch.randn((3, 3, 3, 3, 3, 3)).to(torch.float16),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_reflection_pad3d_2(self):
    args = (
        torch.randint(0, 10, (3, 3, 3, 3, 3, 3)).to(torch.int32),
        [
            1,
            2,
            1,
            2,
            1,
            2,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.reflection_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.reflection_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_relu_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_relu_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_relu_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_relu_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.relu(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.relu, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_remainder_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.remainder.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.remainder.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_remainder_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.remainder.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.remainder.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_remainder_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.remainder.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.remainder.Tensor, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_replication_pad2d_0(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_replication_pad2d_1(self):
    args = (
        torch.randn((3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_replication_pad2d_2(self):
    args = (
        torch.randint(0, 10, (3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_replication_pad3d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_replication_pad3d_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_replication_pad3d_2(self):
    args = (
        torch.randint(0, 10, (1, 3, 2, 10)).to(torch.int32),
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.replication_pad3d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.replication_pad3d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_roll_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.roll(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.roll, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_roll_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.roll(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.roll, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_roll_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.roll(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.roll, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_round_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.round(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.round, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_round_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.round(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.round, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_round_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.round(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.round, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_rsqrt_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.rsqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rsqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_rsqrt_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.rsqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rsqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_rsqrt_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.rsqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rsqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_rsqrt_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.rsqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.rsqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_scalar_tensor_0(self):
    args = (1,)
    kwargs = dict()
    res = torch.ops.aten.scalar_tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scalar_tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_add_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_add(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_add, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_add_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_add(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_add, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_add_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_add(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_add, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_add_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (2, 2)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_add(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_add, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_reduce_two_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
        "sum",
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_reduce.two(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_reduce.two, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_reduce_two_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
        "sum",
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_reduce.two(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_reduce.two, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_scatter_reduce_two_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
        "sum",
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_reduce.two(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_reduce.two, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_reduce_two_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        "sum",
    )
    kwargs = dict()
    res = torch.ops.aten.scatter_reduce.two(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter_reduce.two, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_src_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.src(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.src, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_src_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.src(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.src, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_src_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.src(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.src, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_value_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.value(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.value, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_value_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.value(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.value, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_scatter_value_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        torch.randint(0, 10, (10, 10)).to(torch.int64),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.scatter.value(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.scatter.value, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_int_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.select.int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select.int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_int_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.select.int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select.int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_int_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.select.int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select.int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_int_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.select.int(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select.int, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_scatter_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    res = torch.ops.aten.select_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_scatter_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    res = torch.ops.aten.select_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_select_scatter_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10,)).to(torch.int64),
        1,
        0,
    )
    kwargs = dict()
    res = torch.ops.aten.select_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.select_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sigmoid_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sigmoid(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sigmoid, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sigmoid_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sigmoid(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sigmoid, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sigmoid_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.sigmoid(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sigmoid, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_sigmoid_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.sigmoid(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sigmoid, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sign_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sign(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sign, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sign_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sign(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sign, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sign_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.sign(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sign, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sign_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.sign(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sign, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sin_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sin_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sin_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.sin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_sin_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.sin(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sin, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sinh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sinh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sinh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.sinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_sinh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.sinh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sinh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_scatter_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_scatter_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_scatter_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_scatter_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice_scatter(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice_scatter, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_slice_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.slice.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.slice.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__softmax_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten._softmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._softmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten__softmax_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten._softmax(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten._softmax, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sort_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.sort(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sort, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_sort_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.sort(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sort, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_sort_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.sort(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sort, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_split_with_sizes_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.split_with_sizes(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.split_with_sizes, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_split_with_sizes_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.split_with_sizes(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.split_with_sizes, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_split_with_sizes_2(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            1,
            2,
            3,
            4,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.split_with_sizes(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.split_with_sizes, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_sqrt_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_sqrt_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.sqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_sqrt_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.sqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_sqrt_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.sqrt(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sqrt, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dim_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dim_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dim_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dim_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dim(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dim, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dims_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dims_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dims_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_squeeze_dims_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            0,
            1,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.squeeze.dims(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.squeeze.dims, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sub_Tensor_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.sub.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sub.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sub_Tensor_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.sub.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sub.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sub_Tensor_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.sub.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sub.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sub_Tensor_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.sub.Tensor(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sub.Tensor, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sum_dim_IntList_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.sum.dim_IntList(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sum.dim_IntList, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sum_dim_IntList_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.sum.dim_IntList(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sum.dim_IntList, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_sum_dim_IntList_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.sum.dim_IntList(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sum.dim_IntList, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_sum_dim_IntList_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        None,
    )
    kwargs = dict()
    res = torch.ops.aten.sum.dim_IntList(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.sum.dim_IntList, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_tan_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.tan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_tan_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.tan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_tan_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.tan(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tan, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_tanh_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.tanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_tanh_1(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.tanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_tanh_2(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.tanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.expectedFailure
  def test_aten_tanh_3(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.tanh(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.tanh, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_topk_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.topk(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.topk, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_topk_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.topk(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.topk, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_topk_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.topk(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.topk, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_topk_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
        1,
        False,
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.topk(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.topk, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(len(res) == len(res2))
    for r, r2 in zip(res, res2):
      self.assertTrue(torch.allclose(r, r2.detach().cpu(), atol=1e-3))

  def test_aten_trunc_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.trunc(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.trunc, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_trunc_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.trunc(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.trunc, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_trunc_2(self):
    args = (torch.randint(0, 10, (10, 10)).to(torch.int32),)
    kwargs = dict()
    res = torch.ops.aten.trunc(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.trunc, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_unsqueeze_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.unsqueeze(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.unsqueeze, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_unsqueeze_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.unsqueeze(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.unsqueeze, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_unsqueeze_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.unsqueeze(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.unsqueeze, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_unsqueeze_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        1,
    )
    kwargs = dict()
    res = torch.ops.aten.unsqueeze(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.unsqueeze, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_upsample_bilinear2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.upsample_bilinear2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.upsample_bilinear2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_upsample_bilinear2d_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
        False,
    )
    kwargs = dict()
    res = torch.ops.aten.upsample_bilinear2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.upsample_bilinear2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_upsample_nearest2d_0(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.upsample_nearest2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.upsample_nearest2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_upsample_nearest2d_1(self):
    args = (
        torch.randn((1, 3, 2, 10)).to(torch.float32),
        [
            3,
            20,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.upsample_nearest2d(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.upsample_nearest2d, args,
                                   kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_var_correction_0(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict()
    res = torch.ops.aten.var.correction(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.var.correction, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_var_correction_1(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict()
    res = torch.ops.aten.var.correction(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.var.correction, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_var_correction_2(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict((
        "correction",
        0,
    ))
    res = torch.ops.aten.var.correction(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.var.correction, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_var_correction_3(self):
    args = (torch.randn((10, 10)).to(torch.float32),)
    kwargs = dict((
        "correction",
        0,
    ))
    res = torch.ops.aten.var.correction(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.var.correction, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  @unittest.skip
  def test_aten_var_correction_4(self):
    args = (torch.randn((10, 10)).to(torch.float16),)
    kwargs = dict((
        "correction",
        0,
    ))
    res = torch.ops.aten.var.correction(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.var.correction, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_view_0(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.view(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.view, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_view_1(self):
    args = (
        torch.randn((10, 10)).to(torch.float32),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.view(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.view, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_view_2(self):
    args = (
        torch.randn((10, 10)).to(torch.float16),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.view(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.view, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_view_3(self):
    args = (
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        [
            1,
            100,
        ],
    )
    kwargs = dict()
    res = torch.ops.aten.view(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.view, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_where_self_0(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.where.self(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.where.self, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_where_self_1(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randn((10, 10)).to(torch.float32),
        torch.randn((10, 10)).to(torch.float32),
    )
    kwargs = dict()
    res = torch.ops.aten.where.self(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.where.self, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_where_self_2(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randn((10, 10)).to(torch.float16),
        torch.randn((10, 10)).to(torch.float16),
    )
    kwargs = dict()
    res = torch.ops.aten.where.self(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.where.self, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))

  def test_aten_where_self_3(self):
    args = (
        torch.randn((10, 10)).to(torch.bool),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
        torch.randint(0, 10, (10, 10)).to(torch.int32),
    )
    kwargs = dict()
    res = torch.ops.aten.where.self(*args, **kwargs)
    exported = torch.export.export(torch.ops.aten.where.self, args, kwargs)
    shlo = exported_program_to_stablehlo(exported)
    res2 = shlo(*args, **kwargs)

    self.assertTrue(torch.allclose(res, res2.detach().cpu(), atol=1e-3))


if __name__ == '__main__':
  unittest.main()
