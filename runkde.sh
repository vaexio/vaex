dirname=/home/data/gavi
datfile=$dirname/gaussian3d-1e4.dat
h5file=$dirname/gaussian3d-1e4.hdf5
N=10000
/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 1 1
./src/maxtreeForSpaces density1.pgm
/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 1 2
./src/maxtreeForSpaces density1.pgm
/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 1 3
./src/maxtreeForSpaces density1.pgm

/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 2 1 2
./src/maxtreeForSpaces density.pgm
/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 2 1 3
./src/maxtreeForSpaces density.pgm
/usr/bin/time -f "Time elapsed: %e"  ./src/kerneldensity $N 3 $datfile 2 2 3
./src/maxtreeForSpaces density.pgm
./src/maxtreeForSpaces test.pgm

#display density.pgm&
#/usr/bin/time -f "Time elapsed: %e" python ./bin/kde-epan-hdf5 $h5file x y
#display density2d-py.pgm&
#diff density2d-py.pgm density.pgm