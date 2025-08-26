import torch
import numpy as np
import ase.io.vasp
from ase import Atoms, units
from ase.io import extxyz
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
from ase import Atom

import sys
sys.path.append("/public/home/zyx/LYC/REANN/REANN-stress-realease/reann/ASE/calculators/")
from reann_stress import REANN  # your_module 是包含了 Equi_MPNN 类的模块
#import sys
#sys.path.append("/public/home/zyx/LYC/REANN/Equi-NN-main/ASE/")
#import getneigh as getneigh
# 定义构型文件路径
config_file = "/public/home/zyx/dataset/S.P.Ong/Equi-NN_LYC1845/stress/output/conf.extxyz"  # 替换为你的构型文件路径
f=open("/public/home/zyx/dataset/S.P.Ong/Equi-NN_LYC1845/stress/output/test-stress_ML.dat",'w+')
# 读取构型文件

calc=REANN()

with open(config_file, 'r') as fileobj:
    configuration = extxyz.read_extxyz(fileobj, index=slice(0, 184))

# 定义 计算器
#print(atoms.positions)
#--------------the type of atom--------------
atomtype = ['Cu','Ce','O','C']
#-----------------the device is cpu or gpu( cpu is default)---------------
device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
period=[1,1,1]
#---------------nn file('EANN_PES_DOUBLE.pt' is default in eann,'REANN_PES_DOUBLE.pt' is default in reann)----------------------------
nn = 'EANN_PES_DOUBLE.pt'
#----------------------eann (if you use EANN package )--------------------------------
#atoms.calc = EANN(device=device,atomtype=atomtype,period=period,nn = nn)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
#atoms.calc = REANN(device=device,atomtype=atomtype,period=[1,1,0],nn = 'REANN_PES_DOUBLE.pt')
print(atoms)
dyn = LBFGS(atoms,trajectory='atom2.traj')
dyn.run(fmax=0.1,steps=100)
traj = Trajectory('atom2.traj')
atoms = traj[-1]
print(atoms.get_potential_energy())
ase.io.write('POSCAR-final', atoms, format='vasp', vasp5='True')
#print(atoms.get_potential_energy())
#print(atoms)
#e= atoms.get_potential_energy()
#f = atoms.get_forces()
#print(e,f)



    #def calculate(self,atoms=None, properties=['energy'],
    #              system_changes=all_changes):

    calc = REANN(atomtype, device=device, properties=['energy','force','stress'] nn='REANN_PES_DOUBLE.pt', dtype=torch.float32)

# 遍历每个构型
    for atoms in configuration:
    # 关联计算器
        atoms.set_calculator(calc)
    
        energy = atoms.get_potential_energy(apply_constraint=False)
        force = atoms.get_forces()
    # 计算应力
        stress = atoms.get_stress(voigt=False) / units.GPa  # 转换为 GPa 单位
        #print("Stress:", stress)
        a=stress[0]/(-160.21766)
        b=stress[1]/(-160.21766)
        c=stress[2]/(-160.21766)
        f.write("%18.8f\n" %(a[0]))
        f.write("%18.8f\n" %(a[1]))
        f.write("%18.8f\n" %(a[2]))
        f.write("%18.8f\n" %(b[0]))
        f.write("%18.8f\n" %(b[1]))
        f.write("%18.8f\n" %(b[2]))
        f.write("%18.8f\n" %(c[0]))
        f.write("%18.8f\n" %(c[1]))
        f.write("%18.8f\n" %(c[2]))
        
        #print(a[0])
        #print(b)
        #print(c)
