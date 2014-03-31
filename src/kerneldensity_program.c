#include "kerneldensity.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>


int main(int argc, char *argv[]){
  char *infilename=argv[3];
  FILE *infile = fopen(infilename,"r");
  int numdata=atoi(argv[1]),numcols=atoi(argv[2]), i,j;
  double **data;
  double *minima,*maxima, *prob;
  double gmean,hscale;
  map1d density1d;
  map2d density;
  map3d density3;
  clock_t start;
  struct tms tstruct;
  long tickspersec = sysconf(_SC_CLK_TCK);  
  double musec;
  int dims = 2;
  int k=0,l=1,m=2;
  
  if (argc>4){
    dims = atoi(argv[4]);
  }
  if(argc>5){
    k=atoi(argv[5])-1; 
  }
  if(argc>6){
    l=atoi(argv[6])-1; 
  }
  if(argc>7){
    m=atoi(argv[7])-1; 
  }



  data = (double **)calloc(numcols, sizeof(double*));
  minima= (double *)calloc(numcols,sizeof(double));
  maxima= (double *)calloc(numcols,sizeof(double));
  prob = (double *)calloc(numdata,sizeof(double));

  for (j=0;j<numcols;j++){
    data[j]= (double *)calloc(numdata,sizeof(double));
  }


  printf("reading data\n");
  for (i = 0; i<numdata; i++){
    for (j=0;j<numcols;j++)
      fscanf(infile," %lf",data[j]+i);
    fscanf(infile," \n");
  }
  printf("done reading data\n");
 
  for (j=0;j<numcols;j++){
    findminmax(numdata,data[j],minima+j,maxima+j);
    printf("mimimum[%d] = %f,maximum[%d] = %f \n ",j, minima[j], j, maxima[j] );
  }

	if(dims == 1) {
		init_map1d(&density1d,minima[k],maxima[k],512);
		double sx1 = 2.0*(maxima[k]-minima[k])/cbrt(numdata);

		gmean = comp_data_probs_1d(&density1d, sx1, numdata, data[k], prob);
		adaptive_dens_1d(&density1d, sx1, gmean, numdata, data[k], prob);


		ImagePGMBinWrite1D_map(&density1d,"density1.pgm");
	}
  else if (dims==2){
    init_map2d(&density,minima[k]-0.06125*(maxima[k]-minima[k]),
	       maxima[k]+0.06125*(maxima[k]-minima[k]),512,
	       minima[l]-0.06125*(maxima[l]-minima[l]),
	       maxima[l]+0.06125*(maxima[l]-minima[l]),512);
    
    
    start = times(&tstruct);
  
    hscale = 6.0/sqrt(numdata); 
    
    gmean = comp_data_probs_2d(&density,
			       hscale*(maxima[k]-minima[k]),
			       hscale*(maxima[l]-minima[l]),
			       numdata, data[k], data[l], prob);
    
    printf("gmean: %f\n", gmean);
    comp_density_2d ( &density,hscale*(maxima[k]-minima[k]),
		      hscale*(maxima[l]-minima[l]),gmean,numdata,
		      data[k],data[l],prob);
    
    musec = (double)(times(&tstruct) - start)/((double)tickspersec);
    
    printf("wall-clock time: %f s\n",musec);
    
    ImagePGMBinWrite_map(&density,"density.pgm");
  } else {
    init_map3d(&density3,minima[k]-0.06125*(maxima[k]-minima[k]),
	       maxima[k]+0.06125*(maxima[k]-minima[k]),256,
	       minima[l]-0.06125*(maxima[l]-minima[l]),
	       maxima[l]+0.06125*(maxima[l]-minima[l]),256,
	       minima[m]-0.06125*(maxima[m]-minima[m]),
	       maxima[m]+0.06125*(maxima[m]-minima[m]),256);
    
    
    start = times(&tstruct);
  
    hscale = 8.0/sqrt(numdata); 
    
    gmean = comp_data_probs_3d(&density3,
			       hscale*(maxima[k]-minima[k]),
			       hscale*(maxima[l]-minima[l]),
			       hscale*(maxima[m]-minima[m]),
			       numdata, data[k], data[l], data[m], prob);
    
    
    comp_density_3d ( &density3,
		      hscale*(maxima[k]-minima[k]),
		      hscale*(maxima[l]-minima[l]),
		      hscale*(maxima[m]-minima[m]),
		      gmean,numdata,
		      data[k],data[l],data[m],prob);
    
    musec = (double)(times(&tstruct) - start)/((double)tickspersec);
    
    printf("wall-clock time: %f s\n",musec);
    
    AVSdens3dwrite(4095,&density3,"density.fld");
  }
  return 0;
}



