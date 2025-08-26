class NNMod(Module):
  __parameters__ = []
  __buffers__ = ["initpot", ]
  initpot : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  outputneuron : int
  net : __torch__.torch.nn.modules.container.___torch_mangle_1.Sequential
  def forward(self: __torch__.src.MODEL.NNMod,
    density: Tensor) -> Tensor:
    net = self.net
    return (net).forward(density, )
class ResBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  resblock : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.src.MODEL.ResBlock,
    x: Tensor) -> Tensor:
    resblock = self.resblock
    _0 = torch.add((resblock).forward(x, ), x)
    return _0
