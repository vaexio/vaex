#dirname=/home/data/gavi
dirname=.
#datfile=/home/data/gavi/gaussian2d-1e8.dat
#h5file=/home/data/gavi/gaussian2d-1e3.hdf5
datfile=$dirname/gaussian3d-1e4.dat
h5file=$dirname/gaussian3d-1e4.hdf5
N=10000
options="-d3 -n x,y,z -m 0.2:0.7 -s 0.01,0.01,0.1:0.01,0.1,0.1 -N $N"
/usr/bin/time ./data/gendata.py $options -u 0.5 > $datfile
/usr/bin/time ./data/gendata.py $options -u 0.5 -o $h5file

