class NNMod(Module):
  __parameters__ = []
  __buffers__ = ["initpot", ]
  initpot : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  outputneuron : int
  net : __torch__.torch.nn.modules.container.___torch_mangle_10.Sequential
  def forward(self: __torch__.src.MODEL.___torch_mangle_11.NNMod,
    density: Tensor) -> Tensor:
    net = self.net
    return (net).forward(density, )
