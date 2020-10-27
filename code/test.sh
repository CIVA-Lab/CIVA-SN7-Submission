conda activate solaris
python infer.py $1 
python postproc.py $1 
python solution.py $2 