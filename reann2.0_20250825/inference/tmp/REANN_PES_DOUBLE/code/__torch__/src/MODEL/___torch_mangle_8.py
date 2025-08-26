class ResBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  resblock : __torch__.torch.nn.modules.container.___torch_mangle_7.Sequential
  def forward(self: __torch__.src.MODEL.___torch_mangle_8.ResBlock,
    x: Tensor) -> Tensor:
    resblock = self.resblock
    _0 = torch.add((resblock).forward(x, ), x)
    return _0
