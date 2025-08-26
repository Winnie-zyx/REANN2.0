import torch
import numpy as np
import os
from src.density import *
#from inference.density import *
from inference.get_neigh import *
from src.MODEL import *

class PES(torch.nn.Module):
    def __init__(self,nlinked=1):
        super(PES, self).__init__()
        #========================set the global variable for using the exec=================
        global nblock, nl, dropout_p, table_norm, activate, norbit
        global oc_loop,oc_nblock, oc_nl, oc_dropout_p, oc_table_norm, oc_activate
        global nwave, neigh_atoms, cutoff, nipsin, atomtype
        # global parameters for input_nn
        nblock = 1                    # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
        nl=[128,128]                # NN structure
        dropout_p=[0.0,0.0]       # dropout probability for each hidden layer
        activate = 'Relu_like'
        table_norm= True
        oc_loop = 1
        oc_nl = [128,128]          # neural network architecture   
        oc_nblock = 1
        oc_dropout_p=[0.0,0.0]
        oc_activate = 'Relu_like'
        #========================queue_size sequence for laod data into gpu
        oc_table_norm=True
        norbit= None
        #======================read input_nn==================================
        with open('para/input_nn','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())
        # define the outputneuron of NN
        outputneuron=1
        #======================read input_nn=============================================
        nipsin=2
        cutoff=4.5
        nwave=7
        neigh_atoms=150
        with open('para/input_density','r') as f1:
           while True:
              tmp=f1.readline()
              if not tmp: break
              string=tmp.strip()
              if len(string)!=0:
                  if string[0]=='#':
                     pass
                  else:
                     m=string.split('#')
                     exec(m[0],globals())

        if activate=='Tanh_like':
            from src.activate import Tanh_like as actfun
        else:
            from src.activate import Relu_like as actfun

        if oc_activate=='Tanh_like':
            from src.activate import Tanh_like as oc_actfun
        else:
            from src.activate import Relu_like as oc_actfun        

        dropout_p=np.array(dropout_p)
        oc_dropout_p=np.array(oc_dropout_p)
        maxnumtype=len(atomtype)
        #========================use for read rs/inta or generate rs/inta================
        if 'rs' in globals().keys():
            rs=torch.from_numpy(np.array(rs))
            inta=torch.from_numpy(np.array(inta))
            nwave=rs.shape[1]
        else:
            inta=torch.ones((nwave))
            rs=torch.rand(nwave)*np.sqrt(cutoff)
        #======================for orbital================================
        nipsin+=1
        if not norbit:
            norbit=int(nwave*(nwave+1)/2*nipsin)
        #========================nn structure========================
        nl.insert(0,int(norbit))
        oc_nl.insert(0,int(norbit))
        #================read the periodic boundary condition, element and mass=========
        self.cutoff=cutoff
        self.initpot=0
        #self.density=GetDensity(rs,inta,cutoff,nipsin,norbit,ocmod_list)
        self.density = GetDensity(rs, inta, neigh_atoms, self.initpot, nipsin, cutoff,norbit,nwave=8, emb_nblock=1, emb_nl=[1,8], emb_layernorm=True,oc_loop=3, oc_nblock=1, oc_nl=[108,108], oc_dropout_p=[0.0,0.0],oc_layernorm=True, nblock=1, nl=[108,108], dropout_p=[0.0,0.0],layernorm=True,Dtype=torch.float32)
        #================================================nn module==================================================
        self.neigh_list=Neigh_List(cutoff,nlinked)
    
    def forward(self,period_table,cart,cell,species,mass):
        cart=cart.detach().clone()
        neigh_list, shifts=self.neigh_list(period_table,cart,cell,mass)
        cart.requires_grad_(True)
        density,dist_vec,output=self.density(cart,neigh_list,shifts,species)
        output = output+self.getdensity.initpot
        varene = torch.sum(output)
        grad_dist_vec = torch.autograd.grad([varene,],[dist_vec,],retain_graph=True)[0]
        stress=torch.zeros(1)
        if grad_dist_vec is not None:
            grad_outputs : List[Optional[torch.Tensor]] = [grad_dist_vec]
            grad_cart = torch.autograd.grad([dist_vec],[cart],grad_outputs)[0]
            #grad_cart = torch.autograd.grad([varene],[cart])[0]
            omega=torch.dot(torch.cross(cell[0],cell[1]),cell[2])
            stress=-torch.einsum("ij,ik->jk",grad_dist_vec,dist_vec)/(omega)
            if grad_cart is not None:
                return varene.detach(),stress.detach(),-grad_cart.detach()
