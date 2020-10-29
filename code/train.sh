source activate solaris
mkdir /wdata/train
cp $1 /wdata/train
python train.py /wdata/data/`ls /wdata/data`