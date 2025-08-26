def silu(input: Tensor,
    inplace: bool=False) -> Tensor:
  if inplace:
    _0 = torch.silu_(input)
  else:
    _0 = torch.silu(input)
  return _0
def layer_norm(input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _1 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
  return _1
