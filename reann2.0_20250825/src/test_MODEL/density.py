import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe
import MODEL as *

class GetDensity(torch.nn.Module):
    def __init__(self,rs,inta,cutoff,neigh_atoms,nipsin,norbit,ocmod_list,emb_nblock=1,emb_nl=[1,8,8],emb_layernorm=True,initpot=0,ncontract=64,nwave=8):   ##add atom_species, delete rs and inta, atom_species from inputand transport to torch,tensor([[8.],[1.]])
        super(GetDensity,self).__init__()
        '''
        rs: tensor[nwave] float
        inta: tensor[nwave] float
        nipsin: np.array/list   int
        cutoff: float
        norbit is similar to ncontract, which define in read.py
        '''
        #self.rs=nn.parameter.Parameter(rs)     #rs and inta go to the initbias
        #self.inta=nn.parameter.Parameter(inta)
        ##self.coeffs = torch.ones(max_number, max_number) / 10         #max_number=max_element_number+1,atomtype
        ##self.coeffs = torch.nn.Parameter(self.coeffs)
        self.register_buffer('cutoff', torch.Tensor([cutoff]))          #注册缓冲区
        self.register_buffer('nipsin', torch.tensor([nipsin]))
        self.register_buffer('atom_species', torch.tensor(np.array([[1]])))   ##有问题

        # calculate the angular part 
        npara=[1]
        index_para=torch.tensor([0],dtype=torch.long)
        for i in range(1,nipsin):
            npara.append(np.power(3,i))
            index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i))
        self.register_buffer('index_para',index_para)
        ##nipsin=2: index_para=[0,1,1,1,2,2,2,2,2,2,2,2,2]

        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(len(ocmod_list)+1,nipsin,nwave,ncontract)))     #len(ocmod_list),nipsin+1,nwave,ncontract

        initbias=torch.randn(nwave,dtype=Dtype)/neigh_atoms
        initbias=torch.hstack((initbias,inta,rs))

        #self.params=nn.parameter.Parameter(torch.ones_like(self.rs)/float(neigh_atoms))  #use initbias to replace the para.
        #self.hyper=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.rand(self.rs.shape[1],norbit)).\
        #unsqueeze(0).unsqueeze(0).repeat(len(ocmod_list)+1,nipsin,1,1))     #the shape of rs changed in the new version.

        #embedded nn
        self.emb_neighnn=NNMod(nwave*3,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)          ##emb_nl should be given in input
        initbias=torch.randn(ncontract,dtype=Dtype)/neigh_atoms
        self.emb_centernn=NNMod(ncontract,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)     #维度怎么给进去？


        ocmod=OrderedDict()
        for i in range(ocmod):
            initbias=torch.randn(nwave,dtype=Dtype)
            f_oc="memssage_"+str(i)
            ocmod[f_oc]=NNMod(nwave,oc_nblock,list(oc_nl),oc_dropout_p,Relu_like,initbias=initbias,layernorm=iter_layernorm)
        self.ocmod= torch.nn.ModuleDict(ocmod)
        self.outnn=NNMod(1,nblock,nl,dropout_p,Relu_like,initbias=torch.tensor(np.array([initpot])),layernorm=layernorm)           #initpot is loaded from dataloader,remember load it in train.py

    ##def reset_parameters(self):
        ##torch.nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))


    def gaussian(self,distances):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        distances=distances.view(-1,1)
        #radial=torch.empty((distances.shape[0],self.rs.shape[0]),dtype=distances.dtype,device=distances.device)              #the shape of rs
        radial=torch.exp(self.inta * torch.square(distances - self.rs))
        #for itype in range(self.rs.shape[0]):
        #    mask = (species_ == itype)             #待修改
        #    ele_index = torch.nonzero(mask).view(-1)
        #    if ele_index.shape[0]>0:
        #        part_radial=torch.exp(self.inta[itype:itype+1]*torch.square \
        #        (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))
        #        radial.masked_scatter_(mask.view(-1,1),part_radial)
        return radial
    
    def cutoff_cosine(self,distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*numatom*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)
    
    def angular(self,dist_vec,f_cut):
        # Tensor: dist_vec[neighbour*numatom*nbatch,3]
        # return: angular[neighbour*numatom*nbatch,npara[0]+npara[1]+...+npara[ipsin]]
        totneighbour=dist_vec.shape[0]
        dist_vec=dist_vec.permute(1,0).contiguous()
        orbital=f_cut.view(1,-1)
        angular=torch.empty(self.index_para.shape[0],totneighbour,dtype=f_cut.dtype,device=f_cut.device)
        angular[0]=f_cut
        num=1
        for ipsin in range(1,self.nipsin[0]):
            orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        return angular  
    
    def forward(self,cart,numatoms,species,atom_index,shifts,center_factor,neigh_factor):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        #center_factor 
        #neigh_factor
        """
        #print('density atom_index',atom_index.size(),flush=True)
        tmp_index=torch.arange(numatoms.shape[0],device=cart.device)*cart.shape[1]
        self_mol_index=tmp_index.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)
        #print('density cart',cart.size(),'\n',cart,flush=True)

        #id_config=[[i] for i in range(atom_index.shape[1])]
        #id_config=torch.tensor(id_config).reshape(-1)
        id_config=torch.arange(atom_index.shape[1],device=cart.device).reshape(-1,1).repeat(1,atom_index.shape[2]).reshape(-1)        ##atom_index:[2,batchsize,?]

        cart_=cart.flatten(0,1)    ##cart:[batchsize,numatoms,3], cart_:[batchsize*numatoms,3]
        totnatom=cart_.shape[0]    ##totnatom:[batchsize*numatoms]
        padding_mask=torch.nonzero((shifts.view(-1,3)>-1e10).all(1)).view(-1)
        #print('density shifts',shifts.size(),'\n',shifts,flush=True)
        #print('density padding_mask',padding_mask.size(),'\n',padding_mask,flush=True)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12=(atom_index.view(2,-1)+self_mol_index)[:,padding_mask]
        #print('atom_index12',atom_index12.size(),'\n',atom_index12,flush=True)
        #print('self_mol_index',self_mol_index.size(),'\n',self_mol_index,flush=True)
        selected_id_config=id_config[padding_mask]
        #print('density selected_id_config',selected_id_config.size(),'\n',selected_id_config,flush=True)
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        shift_values=shifts.view(-1,3).index_select(0,padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values
        distances = torch.linalg.norm(dist_vec,dim=-1)
        #dist_vec=dist_vec/distances.view(-1,1)
        species_1=species.reshape(-1,1)    ##species:[batchsize*numatoms],species_1:[batchsize*numatoms,1]
        center_coeff=self.emb_centernn(species_1)
        expand_spec=torch.index_select(species_1,0,atom_index12.view(-1)).view(2,-1,1) #whether be consistent with selected_cart?
        hyper_spec=expand_spec[0]*expand_spec[1]/(expand_spec[0]+expand_spec[1])  #why dispose the species of center and neigh atoms by this way
        neigh_emb=self.emb_neighnn(hyper_spec).T.contiguous()
        #cut_distances=neigh_factor*self.cutoff_cosine(distances)
        species_ = species.index_select(0,atom_index12[1])          #delete
        cut_distances=self.cutoff_cosine(distances)
        nangular=self.angular(dist_vec,cut_distances)
        orbital = oe.contract("ji,ik -> ijk",nangular,\
        self.gaussian(distances),backend="torch")       #gaussian dont need species
        orb_coeff=torch.empty((totnatom,self.rs.shape[0]),dtype=cart.dtype,device=cart.device)  #rs.shape   orb_coeff(dm): [totnatom,nwave]
        mask=(species>-0.5).view(-1)
        #orb_coeff.masked_scatter_(mask.view(-1,1),self.params.index_select(0,species[torch.nonzero(mask).view(-1)]))
        orb_coeff.masked_scatter_(mask.view(-1, 1), self.params)  #use the same params for all species
        weight_orbital=torch.einsum("ijk,ki->ijk",orbital,neigh_emb[:self.nwave]).contiguous()     #weight_orbital:[neigh,lenth,nwave]
        zero_orbital=cart.new_zeros((cart.shape[0],nangular[0],self.nwave),dtype=cart.dtype,device=cart.device)     #angular[1]=1+3=4, zero_orbital:[batchsize*numatoms,lenth,nwave]
        center_orbital=torch.index_add(zero_orbital,0,atom_index12[0],weight_orbital)   #atom_index12:[2,batch*numatoms],center_orbital:[batch*numatoms,lenth,nwave]
        #contracted_coeff== orb_coeff
        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff[0])
        density=torch.einsum("ijm,ijm,im ->im",contracted_orbital,contracted_orbital,center_coeff)
        #density=self.obtain_orb_coeff(0,totnatom,orbital,atom_index12,orb_coeff).view(totnatom,-1)
        iter_coeff=neigh_emb[:self.nwave].T.contiguous()
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            nnout=m(density)
            iter_coeff = iter_coeff + torch.index_select(nnout,0,atom_index12[1])
            density,center_orbital=self.density0(orbital,cut_distances,iter_coeff,atom_index12[0],atom_index12[1],contracted_coeff[ioc_loop+1],zero_orbital,center_orbital,center_coeff)
            #density = self.obtain_orb_coeff(ioc_loop+1,totnatom,orbital,atom_index12,orb_coeff)
        output=self.outnn(density)
        return density,dist_vec,selected_id_config,output
 
    def density0(self,orbital,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,zero_orbital,center_orbital,center_coeff):
        weight_orbital = torch.einsum("ik,ijk -> ijk",iter_coeff,orbital)+torch.einsum("ijk,i->ijk",torch.index_select(center_orbital,0,index_neigh),cut_distances)
        center_orbital=torch.index_add(zero_orbital,0,index_center,weight_orbital)
        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff)
        density=torch.einsum("ijm,ijm,im ->im",contracted_orbital,contracted_orbital,center_coeff)
        return density,center_orbital

