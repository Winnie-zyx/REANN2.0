source ~/.bashrc
# conda environment
conda_env=NequIP

conda activate $conda_env
#use 1 for t=0 or 2 for t>0
python3 /public/home/zyx/reann/REANN-stress/REANN-stress-realease/reann_test2502_stress_add_xjf/reann_test2502_stress_add/trans_pth_bf.py $1
