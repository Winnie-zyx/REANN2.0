#source code comes from
#/public/home/xjf/proj-22-0117/lammps/REANN-github-0408/pes/cal

import numpy as np
import torch
#from gpu_sel import *
import os,sys
import time 

def gpu_sel():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_info')
    memory_gpu=[int(x.split()[2]) for x in open('gpu_info','r').readlines()]
    if memory_gpu:
        gpu_queue=sorted(range(len(memory_gpu)), key=lambda k: memory_gpu[k],reverse=True)
        str_queue=""
        for i in gpu_queue:
            str_queue+=str(i)
            str_queue+=", "
        os.environ['CUDA_VISIBLE_DEVICES']=str_queue
        os.system('rm gpu_info')

#----------------------------------------------------------

path_dat=sys.argv[1]+'/configuration'
path_model='./' #sys.argv[2]+'/'
datatype="float"
#datatype="double"

print(1)

# used for select a unoccupied GPU
gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# same as the atomtype in the file input_density

with os.popen(r'grep -n point {0}'.format(path_dat), 'r') as f:
    data=f.readlines()
#id_lines=[]
#for d in data:
#    id_l=int(d.split(':')[0])-1
#    id_lines.append(id_l)
with open(path_dat) as f:
    configs=f.readlines()
#if len(id_lines)==1:
#    id_lines+=[]
#config=configs[id_lines[0]:id_lines[1]]
config=configs[0:8]
#del config[-1]
#assum that pbc in dat-set will keep same
pbc_arr=list(map(int,config[4].split()[1:4]))
period_table=torch.tensor(pbc_arr,dtype=torch.double,device=device)
print(period_table)
#del config[:5]


#print(atomtype,'\n',mass)
#sys.exit()

#load the serilizable model
if datatype=="double":
    tensor_type = torch.double
    pes=torch.jit.load(path_model+"REANN_PES_DOUBLE.pt")
else:
    tensor_type = torch.float
    pes=torch.jit.load(path_model+"REANN_PES_FLOAT.pt")
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(tensor_type)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)

# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
#period_table=torch.tensor([1,1,1],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
npoint1=0
rmse1=torch.zeros(2,dtype=torch.double,device=device)
rmse=torch.zeros(2,dtype=torch.double,device=device)
mae=torch.zeros(2,dtype=torch.double,device=device)
#with open("/share/home/bjiangch/atnn/data_methane/new_test_validation/test/configuration",'r') as f1:
with open(path_dat,'r') as f1:
    while True:
        string=f1.readline()
        if not string: break
        string=f1.readline()
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
        string=f1.readline()
        numatom=0
        mass=[]
        #atomtype=[]
        atomtype=['H','O']
        species=[]
        cart=[]
        abforce=[]
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart.append(tmp1[0:3])
            abforce.append(tmp1[3:6])
            ele_tmp=tmp[0]
            mass.append(float(tmp[1]))
            if ele_tmp not in atomtype: atomtype.append(ele_tmp)
            species.append(atomtype.index(tmp[0]))
            numatom+=1
        #
        #mass=torch.from_numpy(np.array(mass)).to(device).to(tensor_type)
        mass=torch.tensor(mass,dtype=torch.double,device=device).to(tensor_type)
        abene=float(string.split()[1])
        abene=torch.from_numpy(np.array([abene])).to(tensor_type).to(device)
        species=torch.from_numpy(np.array(species)).to(device)
        cart=torch.from_numpy(np.array(cart)).to(device).to(tensor_type)
        abforce=torch.from_numpy(np.array(abforce)).to(device).to(tensor_type)
        tcell=torch.from_numpy(cell).to(device).to(tensor_type)
        #print('period_table',cart)
        energy,stress,force=pes.forward(period_table,cart,tcell,species,mass)
        energy_t=energy.detach().cpu().to(tensor_type)
        force_t =force.detach().cpu().to(tensor_type)
        print("energy_t ",energy_t)
        print(force_t)
        rmse[0]+=torch.sum(torch.square(energy_t-abene))
        rmse[1]+=torch.sum(torch.square(force_t-abforce))
        mae[0] +=torch.sum(torch.abs(energy_t-abene))
        mae[1] +=torch.sum(torch.abs(force_t-abforce))
        npoint+=1
        #if npoint==1593:break

rmse=rmse.detach().cpu().numpy()
rmse_e=np.sqrt(rmse[0]/npoint)#/23.061
rmse_f=np.sqrt(rmse[1]/npoint/(numatom*3))#/23.061

mae=mae.detach().cpu().numpy()
mae_e=mae[0]/npoint#/23.061
mae_f=mae[1]/npoint/(numatom*3)#/23.061

#print('mae  {0} {1}'.format(mae_e,mae_f),flush=True)
#print('rmse {0} {1}'.format(rmse_e,rmse_f),flush=True)
str_rmse='rmse {0:12.9f} {1:12.9f}'.format(rmse_e/numatom,rmse_f)
str_mae ='mae  {0:12.9f} {1:12.9f}'.format(mae_e/numatom,mae_f)
print(str_rmse,str_mae,flush=True)
