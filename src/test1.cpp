  
#include <malloc.h>
//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "CImg.h"
  using namespace cimg_library;


void findminmax(int numdata, double *data, double *min, double *max){
  int i;

  *max = *min = data[0];
  for (i=1;i<numdata;i++){
    if (data[i]>(*max))
      *max=data[i];
    else
      if (data[i]<(*min))
	*min=data[i];
  }
}

int main(int argc, char *argv[]) {

CImg<unsigned char> img(1000,500,1,3,0);
CImgDisplay main_disp(img,"Draw Parallel Coordinates");
       const unsigned char color[] = { 255,128,64 },color1[] = { 64,128,255 };

int parLine[100][2],i,numpar=6,start=40, end=1000,inc,lengthpar=400;
double **data,*maxima,*minima,max,min;
int numdata=9,numcols=numpar,k=0,m=0,n,j;
char *infilename=argv[1];
FILE *infile = fopen(infilename,"r");

  data = (double **)calloc(numcols, sizeof(double*));
  minima= (double *)calloc(numcols,sizeof(double));
  maxima= (double *)calloc(numcols,sizeof(double));

for (j=0;j<numcols;j++)
    data[j]= (double *)calloc(numdata,sizeof(double));

  printf("reading data....................\n");
  for (i = 0; i<numdata; i++){
    for (j=0;j<numcols;j++)
      { if(j<numcols-1)
         { fscanf(infile," %lf",data[j]+i);
           fscanf(infile,"%*c");
          printf("%lf\t", *(data[j]+i));
         }
        else
           {fscanf(infile," %lf",data[j]+i);
            printf("%lf", *(data[j]+i));
	   }
      }//printf("\n");
      
    fscanf(infile," \n");
  }

  printf("done reading data-----------------\n");   

 for (j=0;j<numcols;j++){
    findminmax(numdata,data[j],minima+j,maxima+j);
    printf("mimimum[%d] = %f,maximum[%d] = %f \n ",j, minima[j], j, maxima[j] );
    /*if(j==0)
       {max=maxiam[j];min=minima[j];}
    else
       {
        if(max<maxima[j])
           max=maxima[j];
        if(min<minima[j])
           min=minima[j]; 
        }*/
  } 

//char min="0.1",max="103.50";
inc=(end-start)/numpar;
/*
parLine[0][0]=40;
parLine[0][1]=40;
parLine[1][0]=40;
parLine[1][1]=300;

parLine[2][0]=40+60;
parLine[2][1]=40;
parLine[3][0]=40+60;
parLine[3][1]=300;

parLine[4][0]=40+60+60;
parLine[4][1]=40;
parLine[5][0]=40+60+60;
parLine[5][1]=300;*/

//Create parallel lines
j=0;
for(i=0;i<=(2*numpar-1);i=i+2)
{
parLine[i][0]=start+j*inc;
parLine[i][1]=start;
parLine[i+1][0]=start+j*inc;
parLine[i+1][1]=lengthpar;
j++;
}


for(i=0;i<numdata;i++)
   
{ for(j=0;j<numcols;j++)
         
            {
	     data[j][i]=(double)start+(((data[j][i]-41.0)/(718.120441 -41.0))*(double)(lengthpar-start));
             printf("%lf\t", *(data[j]+i));
            }
printf("\n");
}
printf("DONE\n");

k=0;m=3;

while (!main_disp.is_closed())
{
//drawing parallellines
  for(i=0;i<=(2*numpar-1);i=i+2)      
{img.draw_line(parLine[i][0],parLine[i][1],parLine[i+1][0],parLine[i+1][1],color).display(main_disp);
	img.draw_text(parLine[i][0],parLine[i][1]-30,"108.50",color,0,13).display(main_disp);
img.draw_text(parLine[i][0],lengthpar+20,"8.50",color,0,13).display(main_disp);
}
//drawing datalines
//for(k=0;k<=(2*numpar-1);k=k+2) 
   for(m=0;m<numcols-1;m++)
         for(i=0;i<numdata;i++)
    img.draw_line(parLine[k][0],data[m][i],parLine[k+2][0],data[m+1][i],color1).display(main_disp);

}
 return 0;
  }
