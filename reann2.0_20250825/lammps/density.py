import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe
from src.MODEL import *
from src.activate import Relu_like,Tanh_like 
#import MODEL as MODEL

class GetDensity(torch.nn.Module):
    def __init__(self,neigh_atoms, nipsin=2, cutoff=4.0, norbit=64, nwave=8, emb_nblock=1, emb_nl=[1,8], emb_layernorm=True,oc_loop=3, oc_nblock=1, oc_nl=[64,64], oc_dropout_p=[0.0,0.0],oc_layernorm=True, nblock=1, nl=[64,64], dropout_p=[0.0,0.0],layernorm=True,initpot=0.0,Dtype=torch.float32):   ##add atom_species, delete rs and inta, atom_species from inputand transport to torch,tensor([[8.],[1.]])
        super(GetDensity,self).__init__()
        '''
        rs: tensor[nwave] float
        inta: tensor[nwave] float
        nipsin: np.array/list   int
        cutoff: float
        norbit is similar to ncontract, which define in read.py
        '''
        self.nwave=nwave
        self.norbit=norbit 
        self.register_buffer('cutoff', torch.Tensor([cutoff]))          #注册缓冲区
        self.register_buffer('nipsin', torch.tensor([nipsin]))
        self.register_buffer('initpot',torch.Tensor([initpot]))
 
        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(oc_loop+1,nipsin,nwave,norbit)))     #len(ocmod_list),nipsin+1,nwave,ncontract

        # calculate the angular part 
        npara1=[1]
        index_para1=torch.tensor([0],dtype=torch.long)
        for i in range(1,nipsin):
            npara1.append(np.power(3,i))
            index_para1=torch.cat((index_para1,torch.ones((npara1[i]),dtype=torch.long)*i))
        
        self.register_buffer('index_para',index_para1)
        #self.register_buffer('index_para',torch.Tensor([0.,1.,1.,1.,2.,2.,2.,2.,2.,2.,2.,2.,2.]))
        ##nipsin=2: index_para=[0,1,1,1,2,2,2,2,2,2,2,2,2]
        
        alpha=torch.ones(nwave,dtype=Dtype)
        rs=(torch.rand(nwave)*np.sqrt(cutoff)).to(Dtype)
        initbias=torch.randn(nwave,dtype=Dtype)/neigh_atoms
        initbias=torch.hstack((initbias,alpha,rs))
        ##rs and inta are given in read.py

        #embedded nn
        self.emb_neighnn=NNMod(nwave*3,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)          ##emb_nl should be given in input
        initbias=torch.randn(norbit,dtype=Dtype)/neigh_atoms
        self.emb_centernn=NNMod(norbit,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)     #维度怎么给进去？

        ocmod=OrderedDict()
        for i in range(oc_loop):
            initbias=torch.randn(nwave,dtype=Dtype)
            f_oc="memssage_"+str(i)
            ocmod[f_oc]=NNMod(nwave,oc_nblock,oc_nl,oc_dropout_p,Relu_like,initbias=initbias,layernorm=oc_layernorm)
        self.ocmod= torch.nn.ModuleDict(ocmod)
        self.outnn=NNMod(1,nblock,nl,dropout_p,Relu_like,initbias=torch.tensor(np.array([1])),layernorm=layernorm)           #initpot is loaded from dataloader,remember load it in train.py

    #def gaussian(self,distances):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        #distances=distances.view(-1,1)
        #radial=torch.empty((distances.shape[0],self.rs.shape[0]),dtype=distances.dtype,device=distances.device)              #the shape of rs
        #radial=torch.exp(self.inta * torch.square(distances - self.rs))
        #for itype in range(self.rs.shape[0]):
        #    mask = (species_ == itype)             #待修改
        #    ele_index = torch.nonzero(mask).view(-1)
        #    if ele_index.shape[0]>0:
        #        part_radial=torch.exp(self.inta[itype:itype+1]*torch.square \
        #        (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))
        #        radial.masked_scatter_(mask.view(-1,1),part_radial)
        #return radial
    
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
        for ipsin in range(1,int(self.nipsin[0])):
            orbital=torch.einsum("ji,ki -> jki",orbital,dist_vec).view(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        return angular  

    def forward(self,cart,atom_index,local_species,neigh_species):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        #center_factor 
        #neigh_factor
        """
        #numatom=cart.shape[0]
        #neigh_species=species.index_select(0,neigh_list[1])
        #selected_cart = cart.index_select(0, neigh_list.view(-1)).view(2, -1, 3)
        #dist_vec = selected_cart[0] - selected_cart[1]-shifts
        #
        nlocal=local_species.shape[0]
        #neigh_species=local_species.index_select(0,neigh_list)
        selected_cart = cart.index_select(0, atom_index.view(-1)).view(2, -1, 3)
        dist_vec = selected_cart[0] - selected_cart[1]
        #
        distances = torch.linalg.norm(dist_vec,dim=-1)   ##distances:[neigh]

        ##
        #species_1=local_species.add(1) #problem
        #num_classes = 118
        #one_hot_encodings1 = torch.nn.functional.one_hot(species_1, num_classes)
        #one_hot_encodings1 = one_hot_encodings1.float()
        #center_coeff=self.emb_centernn(one_hot_encodings1)
        #expand_spec_0=torch.index_select(one_hot_encodings1,0,(atom_index[0]).view(-1)).view(-1,num_classes)  #problem
        ##
        #species_2=neigh_species.add(1) #problem
        #num_classes = 118
        #one_hot_encodings2 = torch.nn.functional.one_hot(species_2, num_classes)
        #one_hot_encodings2 = one_hot_encodings2.float()
        #expand_spec_1=one_hot_encodings2.view(-1,num_classes)
        ##expand_spec_1=torch.index_select(one_hot_encodings2,0,(atom_index[1]).view(-1)).view(-1,num_classes)  #problem
        ##
        ##expand_spec_1=torch.index_select(one_hot_encodings,0,(atom_index[1]).view(-1)).view(-1,num_classes)
        #hyper_spec=expand_spec_0+expand_spec_1
        ##
        species_1=local_species.add(1) #problem
        num_classes = 118
        one_hot_encodings = torch.nn.functional.one_hot(species_1, num_classes)
        one_hot_encodings = one_hot_encodings.float()
        center_coeff=self.emb_centernn(one_hot_encodings)
        expand_spec=torch.index_select(one_hot_encodings,0,(atom_index).view(-1)).view(2,-1,num_classes)  #problem
        hyper_spec=expand_spec[0]+expand_spec[1]

        neigh_emb=self.emb_neighnn(hyper_spec).T.contiguous()    ##[3*nwave,neigh]
        cut_distances=self.cutoff_cosine(distances)
        radial_func=torch.exp(-torch.square(neigh_emb[self.nwave:self.nwave*2]*(distances-neigh_emb[self.nwave*2:])))    ##radial_func:[nwave,neigh]
        nangular=self.angular(dist_vec,cut_distances)   ##[lenth,neigh]  lenth=npara[0]+npara[1]+...+npara[ipsin]
        orbital=torch.einsum("ji,ki -> ijk",nangular,radial_func)
        weight_orbital=torch.einsum("ijk,ki->ijk",orbital,neigh_emb[:self.nwave]).contiguous()     #weight_orbital:[neigh,lenth,nwave]
        #zero_orbital=cart.new_zeros((cart.shape[0],nangular.shape[0],self.nwave),dtype=cart.dtype,device=cart.device)     #nangular.shape[0]=1+3+9=13, zero_orbital:[batchsize*numatoms,lenth,nwave]
        zero_orbital=cart.new_zeros((nlocal,nangular.shape[0],self.nwave),dtype=cart.dtype,device=cart.device)     #nangular.shape[0]=1+3+9=13, zero_orbital:[batchsize*numatoms,lenth,nwave]

        contracted_coeff=self.contracted_coeff[:,self.index_para].contiguous()
        center_orbital=torch.index_add(zero_orbital,0,atom_index[0,:],weight_orbital)   #atom_index12:[2,batch*numatoms],center_orbital:[batch*numatoms,lenth,nwave]
        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff[0])
        ## center_orbital:[batch*numatoms,lenth,nwave], contracted_coeff[0]:[lenth,nwave,norbit]
        density=torch.einsum("ijm,ijm->im",contracted_orbital,contracted_orbital)+center_coeff

        iter_coeff=neigh_emb[:self.nwave].T.contiguous()    ##iter_coeff:[neigh,nwave]
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            nnout=m(density)
            iter_coeff = iter_coeff + torch.index_select(nnout,0,neigh_list)
            density,center_orbital=self.density0(orbital,cut_distances,iter_coeff,atom_index[0],neigh_list,contracted_coeff[ioc_loop+1],zero_orbital,center_orbital,center_coeff)
        #----------------output calculation------------------------------------
        mask_species = (local_species>-0.5)
        output1= self.outnn(density)    ##[numatoms,norbit]
        output = torch.einsum("ij,i->ij",output1,mask_species)
        return dist_vec,output
 
#    def density0(self,orbital,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,zero_orbital,center_orbital,center_coeff):
#        weight_orbital = torch.einsum("ik,ijk -> ijk",iter_coeff,orbital)+torch.einsum("ijk,i->ijk",torch.index_select(center_orbital,0,index_neigh),cut_distances)
#        center_orbital=torch.index_add(zero_orbital,0,index_center,weight_orbital)
#        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff)
#        density=torch.einsum("ijm,ijm->im",contracted_orbital,contracted_orbital)+center_coeff
#        return density,center_orbital

    def log_tensor_memory(tensor, name):
        print(f"{name}: size={tensor.size()}, allocated memory={torch.cuda.memory_allocated()} bytes")
