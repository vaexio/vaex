SUBDIRS = src


echo_prefix:
	echo ${prefix}



data/test.dat: 
	./data/gendata.py -N 100 -s0.01,0.04 -m0.5 -u 0 > $@

density.pgm: data/test.dat
	kerneldensity 100 2 data/test.dat 2 1 2
