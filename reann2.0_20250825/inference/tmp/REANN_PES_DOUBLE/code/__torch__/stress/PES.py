class PES(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  cutoff : float
  getdensity : __torch__.inference.density.GetDensity
  neigh_list : __torch__.inference.get_neigh.Neigh_List
  def forward(self: __torch__.stress.PES.PES,
    period_table: Tensor,
    cart: Tensor,
    cell: Tensor,
    species: Tensor,
    mass: Tensor) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    _0 = uninitialized(Tuple[Tensor, Tensor, Tensor])
    cart0 = torch.clone(torch.detach(cart))
    neigh_list = self.neigh_list
    _1 = (neigh_list).forward(period_table, cart0, cell, mass, )
    neigh_list0, shifts, = _1
    _2 = torch.requires_grad_(cart0)
    getdensity = self.getdensity
    _3 = (getdensity).forward(cart0, neigh_list0, shifts, species, )
    dist_vec, output, = _3
    getdensity0 = self.getdensity
    initpot = getdensity0.initpot
    output0 = torch.add(output, initpot)
    varene = torch.sum(output0)
    _4 = torch.autograd.grad([varene], [dist_vec], None, True)
    grad_dist_vec = _4[0]
    _5 = torch.__isnot__(grad_dist_vec, None)
    if _5:
      grad_dist_vec0 = unchecked_cast(Tensor, grad_dist_vec)
      grad_outputs = annotate(List[Optional[Tensor]], [grad_dist_vec0])
      _8 = torch.autograd.grad([dist_vec], [cart0], grad_outputs)
      grad_cart = _8[0]
      _9 = torch.cross(torch.select(cell, 0, 0), torch.select(cell, 0, 1))
      omega = torch.dot(_9, torch.select(cell, 0, 2))
      _10 = torch.einsum("ij,ik->jk", [grad_dist_vec0, dist_vec])
      stress = torch.div(torch.neg(_10), omega)
      if torch.__isnot__(grad_cart, None):
        grad_cart0 = unchecked_cast(Tensor, grad_cart)
        _13 = torch.detach(varene)
        _14 = torch.detach(stress)
        _15 = torch.neg(torch.detach(grad_cart0))
        _11, _12 = True, (_13, _14, _15)
      else:
        _11, _12 = False, _0
      _6, _7 = _11, _12
    else:
      _6, _7 = False, _0
    if _6:
      _16 : Optional[Tuple[Tensor, Tensor, Tensor]] = _7
    else:
      _16 = None
    return _16
