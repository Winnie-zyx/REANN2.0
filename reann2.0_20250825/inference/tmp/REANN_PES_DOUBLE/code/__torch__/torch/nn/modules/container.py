class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    input: Tensor) -> Tensor:
    return input
  def __len__(self: __torch__.torch.nn.modules.container.Sequential) -> int:
    return 0
class ModuleDict(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : NoneType
  memssage_0 : __torch__.src.MODEL.___torch_mangle_11.NNMod
  memssage_1 : __torch__.src.MODEL.___torch_mangle_11.NNMod
  def __len__(self: __torch__.torch.nn.modules.container.ModuleDict) -> int:
    return 2
  def __contains__(self: __torch__.torch.nn.modules.container.ModuleDict,
    key: str) -> bool:
    _0 = torch.__contains__(["memssage_0", "memssage_1"], key)
    return _0
