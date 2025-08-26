class Relu_like(Module):
  __parameters__ = ["alpha", "beta", ]
  __buffers__ = []
  alpha : Tensor
  beta : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  silu : __torch__.torch.nn.modules.activation.SiLU
  def forward(self: __torch__.src.activate.Relu_like,
    x: Tensor) -> Tensor:
    alpha = self.alpha
    silu = self.silu
    beta = self.beta
    _0 = (silu).forward(torch.mul(x, beta), )
    return torch.mul(alpha, _0)
