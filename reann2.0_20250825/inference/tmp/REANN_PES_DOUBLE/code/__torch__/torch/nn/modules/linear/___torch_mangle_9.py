class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : NoneType
  in_features : Final[int] = 64
  out_features : Final[int] = 8
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_9.Linear,
    input: Tensor) -> Tensor:
    weight = self.weight
    bias = self.bias
    return torch.linear(input, weight, bias)
