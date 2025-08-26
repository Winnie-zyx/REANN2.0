class GetDensity(Module):
  __parameters__ = ["contracted_coeff", ]
  __buffers__ = ["cutoff", "nipsin", "initpot", "index_para", ]
  contracted_coeff : Tensor
  cutoff : Tensor
  nipsin : Tensor
  initpot : Tensor
  index_para : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  nwave : int
  norbit : int
  emb_neighnn : __torch__.src.MODEL.NNMod
  emb_centernn : __torch__.src.MODEL.___torch_mangle_4.NNMod
  ocmod : __torch__.torch.nn.modules.container.ModuleDict
  outnn : __torch__.src.MODEL.___torch_mangle_14.NNMod
  def forward(self: __torch__.inference.density.GetDensity,
    cart: Tensor,
    neigh_list: Tensor,
    shifts: Tensor,
    species: Tensor) -> Tuple[Tensor, Tensor]:
    _0 = torch.index_select(cart, 0, torch.view(neigh_list, [-1]))
    selected_cart = torch.view(_0, [2, -1, 3])
    _1 = torch.sub(torch.select(selected_cart, 0, 0), torch.select(selected_cart, 0, 1))
    dist_vec = torch.sub(_1, shifts)
    distances = torch.linalg_norm(dist_vec, None, [-1])
    species_1 = torch.add(species, 1)
    one_hot_encodings = torch.one_hot(species_1, 118)
    one_hot_encodings0 = torch.to(one_hot_encodings, 6)
    emb_centernn = self.emb_centernn
    center_coeff = (emb_centernn).forward(one_hot_encodings0, )
    _2 = torch.index_select(one_hot_encodings0, 0, torch.view(neigh_list, [-1]))
    expand_spec = torch.view(_2, [2, -1, 118])
    hyper_spec = torch.add(torch.select(expand_spec, 0, 0), torch.select(expand_spec, 0, 1))
    emb_neighnn = self.emb_neighnn
    _3 = torch.numpy_T((emb_neighnn).forward(hyper_spec, ))
    neigh_emb = torch.contiguous(_3)
    cut_distances = (self).cutoff_cosine(distances, )
    nwave = self.nwave
    nwave0 = self.nwave
    _4 = torch.slice(neigh_emb, 0, nwave, torch.mul(nwave0, 2))
    nwave1 = self.nwave
    _5 = torch.slice(neigh_emb, 0, torch.mul(nwave1, 2))
    _6 = torch.mul(_4, torch.sub(distances, _5))
    radial_func = torch.exp(torch.neg(torch.square(_6)))
    nangular = (self).angular(dist_vec, cut_distances, )
    orbital = torch.einsum("ji,ki -> ijk", [nangular, radial_func])
    nwave2 = self.nwave
    _7 = torch.slice(neigh_emb, 0, None, nwave2)
    _8 = torch.einsum("ijk,ki->ijk", [orbital, _7])
    weight_orbital = torch.contiguous(_8)
    _9 = (torch.size(cart))[0]
    _10 = (torch.size(nangular))[0]
    nwave3 = self.nwave
    _11 = ops.prim.dtype(cart)
    _12 = ops.prim.device(cart)
    zero_orbital = torch.new_zeros(cart, [_9, _10, nwave3], dtype=_11, layout=None, device=_12)
    contracted_coeff = self.contracted_coeff
    index_para = self.index_para
    _13 = torch.slice(contracted_coeff)
    _14 = annotate(List[Optional[Tensor]], [None, index_para])
    contracted_coeff0 = torch.contiguous(torch.index(_13, _14))
    center_orbital = torch.index_add(zero_orbital, 0, torch.select(neigh_list, 0, 0), weight_orbital)
    _15 = torch.select(contracted_coeff0, 0, 0)
    contracted_orbital = torch.einsum("ijk,jkm->ijm", [center_orbital, _15])
    _16 = [contracted_orbital, contracted_orbital, center_coeff]
    density = torch.einsum("ijm,ijm,im ->im", _16)
    nwave4 = self.nwave
    _17 = torch.slice(neigh_emb, 0, None, nwave4)
    iter_coeff = torch.contiguous(torch.numpy_T(_17))
    ocmod = self.ocmod
    memssage_0 = ocmod.memssage_0
    memssage_1 = ocmod.memssage_1
    nnout = (memssage_0).forward(density, )
    _18 = torch.index_select(nnout, 0, torch.select(neigh_list, 0, 1))
    iter_coeff0 = torch.add(iter_coeff, _18)
    _19 = torch.select(neigh_list, 0, 0)
    _20 = torch.select(neigh_list, 0, 1)
    _21 = torch.select(contracted_coeff0, 0, 1)
    _22 = (self).density0(orbital, cut_distances, iter_coeff0, _19, _20, _21, zero_orbital, center_orbital, center_coeff, )
    density0, center_orbital0, = _22
    nnout0 = (memssage_1).forward(density0, )
    _23 = torch.index_select(nnout0, 0, torch.select(neigh_list, 0, 1))
    iter_coeff1 = torch.add(iter_coeff0, _23)
    _24 = torch.select(neigh_list, 0, 0)
    _25 = torch.select(neigh_list, 0, 1)
    _26 = torch.select(contracted_coeff0, 0, 2)
    _27 = (self).density0(orbital, cut_distances, iter_coeff1, _24, _25, _26, zero_orbital, center_orbital0, center_coeff, )
    density1, center_orbital1, = _27
    mask_species = torch.gt(species, -0.5)
    outnn = self.outnn
    output1 = (outnn).forward(density1, )
    output = torch.einsum("ij,i->ij", [output1, mask_species])
    return (dist_vec, output)
  def cutoff_cosine(self: __torch__.inference.density.GetDensity,
    distances: Tensor) -> Tensor:
    cutoff = self.cutoff
    _28 = torch.mul(torch.reciprocal(cutoff), 3.1415926535897931)
    _29 = torch.cos(torch.mul(distances, _28))
    _30 = torch.square(torch.add(torch.mul(_29, 0.5), 0.5))
    return _30
  def angular(self: __torch__.inference.density.GetDensity,
    dist_vec: Tensor,
    f_cut: Tensor) -> Tensor:
    totneighbour = (torch.size(dist_vec))[0]
    dist_vec0 = torch.contiguous(torch.permute(dist_vec, [1, 0]))
    orbital = torch.view(f_cut, [1, -1])
    index_para = self.index_para
    _31 = (torch.size(index_para))[0]
    _32 = ops.prim.dtype(f_cut)
    _33 = ops.prim.device(f_cut)
    angular = torch.empty([_31, totneighbour], dtype=_32, layout=None, device=_33)
    _34 = torch.copy_(torch.select(angular, 0, 0), f_cut)
    nipsin = self.nipsin
    _35 = torch.__range_length(1, int(torch.select(nipsin, 0, 0)), 1)
    num = 1
    orbital0 = orbital
    for _36 in range(_35):
      _37 = torch.einsum("ji,ki -> jki", [orbital0, dist_vec0])
      orbital1 = torch.view(_37, [-1, totneighbour])
      _38 = torch.add(num, (torch.size(orbital1))[0])
      _39 = torch.copy_(torch.slice(angular, 0, num, _38), orbital1)
      num0 = torch.add(num, (torch.size(orbital1))[0])
      num, orbital0 = num0, orbital1
    return angular
  def density0(self: __torch__.inference.density.GetDensity,
    orbital: Tensor,
    cut_distances: Tensor,
    iter_coeff: Tensor,
    index_center: Tensor,
    index_neigh: Tensor,
    contracted_coeff: Tensor,
    zero_orbital: Tensor,
    center_orbital: Tensor,
    center_coeff: Tensor) -> Tuple[Tensor, Tensor]:
    _40 = torch.einsum("ik,ijk -> ijk", [iter_coeff, orbital])
    _41 = torch.index_select(center_orbital, 0, index_neigh)
    _42 = torch.einsum("ijk,i->ijk", [_41, cut_distances])
    weight_orbital = torch.add(_40, _42)
    center_orbital2 = torch.index_add(zero_orbital, 0, index_center, weight_orbital)
    contracted_orbital = torch.einsum("ijk,jkm->ijm", [center_orbital2, contracted_coeff])
    _43 = [contracted_orbital, contracted_orbital, center_coeff]
    density = torch.einsum("ijm,ijm,im ->im", _43)
    return (density, center_orbital2)
