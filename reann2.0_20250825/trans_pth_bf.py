##
##load REANN.pth, and then change the torch-script in .pt from the initial code to the changed code 

import sys

mod=sys.argv[1]
if mod=='0':
    import stress.script_PES as PES_save
if mod=='1':
    #import stress.script_PES as PES_Normal
    import lammps.script_PES as PES_save
if mod=='2':
    #import stress.script_PES as PES_Normal
    import lammps_REANN.script_PES as PES_save
   
PES_save.jit_pes()
#PES_Normal.jit_pes()
#PES_Lammps.jit_pes()
