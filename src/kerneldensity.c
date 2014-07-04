#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MALLOC_H // if it doesn't exists, stdlib.h should get all we need
#include <malloc.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>
#include "avs_io.h"
#include "kerneldensity.h"
#include <getopt.h>
#include <string.h>

#define PI 3.141592654


typedef struct { double (*Func)( double sqr_dist);
                 double normalization,
                       range;
               } Kernel;


void init_map2d ( map2d *map,
                  double x_min,
		  double x_max,
                  int   x_bins,
                  double y_min,
		  double y_max,
                  int   y_bins)
{ short i;
  map->x_min=x_min;
  map->x_max=x_max;
  map->x_bins=x_bins;
  map->y_min=y_min;
  map->y_max=y_max;
  map->y_bins=y_bins;
  map->data_max=0.0;
  map->data_min=0.0;

  map->map=(double **) calloc(y_bins,sizeof(double *));
  for (i=0;i<y_bins;i++)
    map->map[i]=(double *) calloc(x_bins,sizeof(double));
}


void init_map3d ( map3d *map,
                  double x_min,
		  double x_max,
                  int   x_bins,
                  double y_min,
		  double y_max,
                  int   y_bins,
                  double z_min,
		  double z_max,
                  int   z_bins)
{ short i,j;
  map->x_min=x_min;
  map->x_max=x_max;
  map->x_bins=x_bins;

  map->y_min=y_min;
  map->y_max=y_max;
  map->y_bins=y_bins;

  map->z_min=z_min;
  map->z_max=z_max;
  map->z_bins=z_bins;

  printf("x_min=%f,x_max=%f, \n",map->x_min,map->x_max);
  printf("y_min=%f,y_max=%f, \n",map->y_min,map->y_max);
  printf("z_min=%f,z_max=%f, \n",map->z_min,map->z_max);

  map->data_max=0.0;
  map->data_min=0.0;

  map->map=(double ***) calloc(z_bins,sizeof(double **));
  for (j=0;j<z_bins;j++){
    map->map[j]=(double **) calloc(z_bins,sizeof(double *));
    
    for (i=0;i<y_bins;i++)
      map->map[j][i]=(double *) calloc(x_bins,sizeof(double));
  }
}


void init_regress_map2d ( regress_map2d *map,
			  double x_min,
			  double x_max,
                          int   x_bins,
			  double y_min,
			  double y_max,
                          int   y_bins )
{ short i;
  map->x_min=x_min;
  map->x_max=x_max;
  map->x_bins=x_bins;
  map->y_min=y_min;
  map->y_max=y_max;
  map->y_bins=y_bins;
  map->data_max=0.0;
  map->data_min=0.0;

  map->dens_map=(double **) calloc(y_bins,sizeof(double *));
  map->z_map=(double **) calloc(y_bins,sizeof(double *));
  for (i=0;i<y_bins;i++)
    { map->dens_map[i]=(double *) calloc(x_bins,sizeof(double));
      map->z_map[i]=(double *) calloc(x_bins,sizeof(double));
    }
}

void init_map1d ( map1d *map,
                  double x_min,
		  double x_max,
                  int   bins)
{ short i;
  map->x_min=x_min;
  map->x_max=x_max;
  map->bins=bins;
  map->data_max=0.0;
  map->data_min=0.0;

  map->map=(double *) calloc(bins,sizeof(double));
  for (i=0;i<bins;i++)
    map->map[i]=0.0;
}


void init_regress_map1d ( regress_map1d *map,
			  double x_min,
			  double x_max,
                          int           bins)
{ short i;
  map->x_min=x_min;
  map->x_max=x_max;
  map->bins=bins;
  map->data_max=0.0;
  map->data_min=0.0;

  map->x_map=(double *) calloc(bins,sizeof(double));
  map->y_map=(double *) calloc(bins,sizeof(double));
  map->sig_map=(double *) calloc(bins,sizeof(double));
  for (i=0;i<bins;i++)
    { map->x_map[i]=0.0;
      map->y_map[i]=0.0;
      map->sig_map[i]=0.0;
    }
}

void exit_map2d ( map2d *map )
{ short i;
  for (i=0;i<map->y_bins;i++)
    free(map->map[i]);
  free(map->map);
}

void exit_map3d ( map3d *map )
{ short i,j;
  for (i=0;i<map->z_bins;i++){
    for (j=0;j<map->y_bins;j++){
      free(map->map[i][j]);
    }
    free(map->map[i]);
  }
  free(map->map);
}

void exit_regress_map2d ( regress_map2d *map )
{ short i;
  for (i=0;i<map->y_bins;i++)
    { free(map->dens_map[i]);
      free(map->z_map[i]);
    }
  free(map->dens_map);
  free(map->z_map);
}

void exit_map1d ( map1d *map )
{
  free(map->map);
}

void exit_regress_map1d ( regress_map1d *map )
{
  free(map->x_map);
  free(map->y_map);
  free(map->sig_map);
}


void clean_regress_map1d ( regress_map1d *d,
                           double min_dens   )
{ int i, valid=0;
  double x=d->x_min,
        xstep=(d->x_max-
                d->x_min)/(d->bins-1);
  for (i=0;i<d->bins;i++,x+=xstep)
    if (d->x_map[i]>min_dens)
      { d->y_map[valid]=(d->y_map[i])/(d->x_map[i]);
        d->sig_map[valid]=sqrt( (d->sig_map[i])/(d->x_map[i]) -
                                (d->y_map[valid])*(d->y_map[valid])  );
        d->x_map[valid]=x;
        if (d->data_max<d->y_map[valid])
          d->data_max=d->y_map[valid];
        if (d->data_min>d->y_map[valid])
          d->data_min=d->y_map[valid];
        valid++;
      }
  d->validbins=valid;
}

void clean_regress_map2d ( regress_map2d *d,
                           double min_dens   )

{ int i,j;
  for (j=0;j<d->y_bins;j++)
    for (i=0;i<d->x_bins;j++)
      if (d->dens_map[j][i]>min_dens)
        { d->z_map[j][i]/=(d->dens_map[j][i]);
          if (d->data_max<d->z_map[j][i])
            d->data_max=d->z_map[j][i];
          if (d->data_min>d->z_map[j][i])
            d->data_min=d->z_map[j][i];
        }
      else
        d->z_map[j][i]=0;
}


void add_one_point_epan ( double       x_val,
                          double       y_val,
                          map2d       *d,
                          double       hx,
                          double       hy    )
/* 
   Add one point using Epanechnikov kernel, without normalization 
   Used for fixed kernel density estimates, which can be post-normalized
   for added speed
*/
 
{ short x,y,x_start;
  double x0,
        cur_y,
        x_step1=(d->x_max- d->x_min)/(hx*(d->x_bins-1)),
        y_step=(d->y_max-
                d->y_min)/(hy*(d->y_bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  cur_y=( (d->y_min)-y_val)/hy;
  if (cur_y<(-1))
    { y=(int)((double)(fabs(cur_y)-1)/y_step);
      cur_y+=y*y_step;
    }
  else
    y=0;
  x0=( (d->x_min)-x_val)/hx;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  while (y<d->y_bins && cur_y<=1)
    {
      cx_xs=x_step1;
      x=x_start;
      dist=cur_y*cur_y+x0*x0;
      while (cx_xs<0 && dist>1)
        { dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      while ( x<d->x_bins && dist<=1)
        { (d->map[y][x])+= (1-dist);   /* no normalization for h */
          dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      cur_y+=y_step;
      y++;
    }
}

void add_one_point_epan_3d ( double       x_val,
			     double       y_val,
			     double       z_val,
			     map3d       *d,
			     double       hx,
			     double       hy,
			     double       hz )
/* 
   Add one point using Epanechnikov kernel, without normalization 
   Used for fixed kernel density estimates, which can be post-normalized
*/
{ short x,y,z,x_start,y_start;
  double 
    x0,y0, cur_y, cur_z,
    x_step1=(d->x_max - d->x_min)/(hx*(d->x_bins-1)),
    y_step=(d->y_max - d->y_min)/(hy*(d->y_bins-1)),
    z_step=(d->z_max - d->z_min)/(hz*(d->z_bins-1)),
    x_step2=x_step1*x_step1,
    cx_xs,
    dist;
  
  cur_z=( (d->z_min)-z_val)/hz;
  if (cur_z<(-1)){
    z=(int)((double)(fabs(cur_z)-1)/z_step);
    cur_z+=z*z_step;
  }
  else
    z=0;

  y0=( (d->y_min)-y_val)/hy;
  if (y0<(-1)){
    y_start=(int)((double)(fabs(y0)-1)/y_step);
    y0+=y_start*y_step;
  }
  else
    y_start=0;

  x0=( (d->x_min)-x_val)/hx;
  if (x0<(-1)){
    x_start=(int)((double)(fabs(x0)-1)/x_step1);
    x0+=x_start*x_step1;
  }
  else
    x_start=0;

  x_step1*=x0;

  while (z<d->z_bins && cur_z<=1){
    cur_y = y0;
    y=y_start;
    while (y<d->y_bins && cur_y<=1){
      cx_xs=x_step1;
      x=x_start;
      dist=cur_z*cur_z+cur_y*cur_y+x0*x0;
      while (cx_xs<0 && dist>1){
	dist+=(cx_xs+cx_xs+x_step2);
	cx_xs+=x_step2;
	x++;
      }
      while ( x<d->x_bins && dist<=1){
	(d->map[z][y][x])+= (1-dist);   /* no normalization for h */
	dist+=(cx_xs+cx_xs+x_step2);
	cx_xs+=x_step2;
	x++;
      }
      cur_y+=y_step;
      y++;
    }
    cur_z+=z_step;
    z++;
  }
}

void add_one_point_epan2 ( double       x_val,
                           double       y_val,
                           map2d       *d,
                           double       hx,
                           double       hy    )
{ short x,y,x_start;
  double x0,
        h2=hx*hy,
        cur_y,
        x_step1=(d->x_max-
                d->x_min)/(hx*(d->x_bins-1)),
        y_step=(d->y_max-
                d->y_min)/(hy*(d->y_bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  cur_y=( (d->y_min)-y_val)/hy;
  if (cur_y<(-1))
    { y=(int)((double)(fabs(cur_y)-1)/y_step);
      cur_y+=y*y_step;
    }
  else
    y=0;
  x0=( (d->x_min)-x_val)/hx;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  while (y<d->y_bins && cur_y<=1)
    {
      cx_xs=x_step1;
      x=x_start;
      dist=cur_y*cur_y+x0*x0;
      while (cx_xs<0 && dist>1)
        { dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      while ( x<d->x_bins && dist<=1)
        { (d->map[y][x])+= ((1-dist)/h2); /* normalized for h */
          dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      cur_y+=y_step;
      y++;
    }
}

void add_one_point_epan2_3d ( double       x_val,
			      double       y_val,
			      double       z_val,
			      map3d       *d,
			      double       hx,
			      double       hy,
			      double       hz )
/* 
   Add one point using Epanechnikov kernel, with partial normalization
   (hx*hy*hz)    
   Used for adaptive kernel density estimates, which can be post-normalized
   for number of data and kernel shape, but not for bandwidth.
*/
{ short x,y,z,x_start,y_start;
  double 
    h3=hx*hy*hz,
    x0,y0, cur_y, cur_z,
    x_step1=(d->x_max - d->x_min)/(hx*(d->x_bins-1)),
    y_step=(d->y_max - d->y_min)/(hy*(d->y_bins-1)),
    z_step=(d->z_max - d->z_min)/(hz*(d->z_bins-1)),
    x_step2=x_step1*x_step1,
    cx_xs,
    dist;
  
  cur_z=( (d->z_min)-z_val)/hz;
  if (cur_z<(-1)){
    z=(int)((double)(fabs(cur_z)-1)/z_step);
    cur_z+=z*z_step;
  }
  else
    z=0;

  y0=( (d->y_min)-y_val)/hy;
  if (y0<(-1)){
    y_start=(int)((double)(fabs(y0)-1)/y_step);
    y0+=y_start*y_step;
  }
  else
    y_start=0;

  x0=( (d->x_min)-x_val)/hx;
  if (x0<(-1)){
    x_start=(int)((double)(fabs(x0)-1)/x_step1);
    x0+=x_start*x_step1;
  }
  else
    x_start=0;

  x_step1*=x0;

  while (z<d->z_bins && cur_z<=1){
    cur_y = y0;
    y=y_start;
    while (y<d->y_bins && cur_y<=1){
      cx_xs=x_step1;
      x=x_start;
      dist=cur_z*cur_z+cur_y*cur_y+x0*x0;
      while (cx_xs<0 && dist>1){
	dist+=(cx_xs+cx_xs+x_step2);
	cx_xs+=x_step2;
	x++;
      }
      while ( x<d->x_bins && dist<=1){
	(d->map[z][y][x])+= (1-dist)/h3;   /* normalization for h */
	dist+=(cx_xs+cx_xs+x_step2);
	cx_xs+=x_step2;
	x++;
      }
      cur_y+=y_step;
      y++;
    }
    cur_z+=z_step;
    z++;
  }
}

void add_one_point_1d ( double       x_val,
                        map1d       *d,
                        double       h    )
{ short x,x_start;
  double x0,
        x_step1=(d->x_max-
                d->x_min)/(h*(d->bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  x0=( (d->x_min)-x_val)/h;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  cx_xs=x_step1;
  x=x_start;
  dist=x0*x0;
  while (cx_xs<0 && dist>1)
    { dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
  while ( x<d->bins && dist<=1)
    { (d->map[x])+= (1-dist);
      dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
}


void add_one_point_epan1d ( double       x_val,
                            map1d       *d,
                            double       h    )
{ short x,x_start;
  double x0,
        x_step1=(d->x_max-
                d->x_min)/(h*(d->bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  x0=( (d->x_min)-x_val)/h;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  cx_xs=x_step1;
  x=x_start;
  dist=x0*x0;
  while (cx_xs<0 && dist>1)
    { dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
  while ( x<d->bins && dist<=1)
    { (d->map[x])+= (1-dist)/h;
      dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
}


void add_one_point_regress_1d ( double         x_val,
                                double         y_val,
                                regress_map1d *d,
                                double         h      )
{ short x,x_start;
  double x0,
        x_step1=(d->x_max-
                 d->x_min)/(h*(d->bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  x0=( (d->x_min)-x_val)/h;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  cx_xs=x_step1;
  x=x_start;
  dist=x0*x0;
  while (cx_xs<0 && dist>1)
    { dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
  while ( x<d->bins && dist<=1)
    {
      (d->x_map[x])+= (1-dist)/h;
      (d->y_map[x])+= y_val*(1-dist)/h;
      (d->sig_map[x])+= y_val*y_val*(1-dist)/h;
      dist+=(cx_xs+cx_xs+x_step2);
      cx_xs+=x_step2;
      x++;
    }
}

void add_one_point_regress_2d ( double         x_val,
                                double         y_val,
                                double         z_val,
                                regress_map2d *d,
                                double         hx,
                                double         hy    )
{ short x,y,x_start;
  double x0,
        h2=hx*hy,
        cur_y,
        x_step1=(d->x_max-
                d->x_min)/(hx*(d->x_bins-1)),
        y_step=(d->y_max-
                d->y_min)/(hy*(d->y_bins-1)),
        x_step2=x_step1*x_step1,
        cx_xs,
        dist;
  cur_y=( (d->y_min)-y_val)/hy;
  if (cur_y<(-1))
    { y=(int)((double)(fabs(cur_y)-1)/y_step);
      cur_y+=y*y_step;
    }
  else
    y=0;
  x0=( (d->x_min)-x_val)/hx;
  if (x0<(-1))
    { x_start=(int)((double)(fabs(x0)-1)/x_step1);
      x0+=x_start*x_step1;
    }
  else
    x_start=0;
  x_step1*=x0;
  while (y<d->y_bins && cur_y<=1)
    {
      cx_xs=x_step1;
      x=x_start;
      dist=cur_y*cur_y+x0*x0;
      while (cx_xs<0 && dist>1)
        { dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      while ( x<d->x_bins && dist<=1)
        { (d->dens_map[y][x])+= ((1-dist)/h2); /* normalized for h */
          (d->z_map[y][x])+= (z_val*(1-dist)/h2); /* normalized for h */
          dist+=(cx_xs+cx_xs+x_step2);
          cx_xs+=x_step2;
          x++;
        }
      cur_y+=y_step;
      y++;
    }
}


void init_contours ( contour_rec *contours,
                     int         num_contours,
                     double       min,
                     double       max,
                     double       log_radix,
                     int         min_col,
                     int         max_col       )
{ int i;
  contours->num_contours=num_contours;
  contours->levels=calloc(num_contours,sizeof(contours->levels[0]));
  contours->colours=calloc(num_contours,sizeof(contours->colours[0]));
  for (i=0;i<num_contours;i++)
    { contours->colours[i]=min_col+i*(max_col-min_col)/(num_contours-1);
      contours->levels[i]=min+(double)i*(max-min)/(double)(num_contours-1);
    }
  if ( log_radix!=0 )
    { contours->levels[num_contours-1]=max;
      for (i=num_contours-2; i>=0; i--)
        contours->levels[i]=contours->levels[i+1]/log_radix;
    }
}

void exit_contours ( contour_rec *contours )
{ free(contours->levels);
  free(contours->colours);
}



double prob_from_map1d(  map1d *map,
                         double x,
                         double xstep )       
{
  int xbin=(int)((x-map->x_min)/xstep);
  double xratio=(x-map->x_min-(xstep*xbin))/xstep;
  return map->map[xbin]+( map->map[xbin+1]
                                    -map->map[xbin]  )*xratio;
}




double comp_data_probs_1d ( map1d  *density,
                            double win_width,
                            int    num_data,
			    double *xdata,
			    double *prob)
{
  int i,k;
  double normalization = 1/( (double)num_data*win_width*4.0/3.0 ), 
         gmean = 0.0;
  double xstep = (density->x_max - density->x_min)/(double)(density->bins - 1);
  for (i=0; i<num_data; i++)
	  if((xdata[i] >= density->x_min) && (xdata[i] <= density->x_max))
		add_one_point_1d(xdata[i],density,win_width);
	  else
		  num_data--;
  density->data_max  =density->map[0]*normalization;
  density->data_min = density->map[0]*normalization;
  for (k=0; k<density->bins; k++)
    {
      density->map[k] = density->map[k]*normalization;
      if (density->map[k]>density->data_max) 
        density->data_max=density->map[k];
      else
        if (density->map[k]< density->data_min)
          density->data_min = density->map[k];
    }

  for (i=0; i<num_data; i++)
    {
		if((xdata[i] >= density->x_min) && (xdata[i] <= density->x_max)) {
			prob[i]=prob_from_map1d(density,xdata[i],xstep);
			gmean += log(prob[i]);
		}
    }

  for (k=0; k<density->bins; k++)
    density->map[k]=0;

  return exp(gmean/num_data);
}


void adaptive_dens_1d(map1d *density, 
		      double win_width,
		      double gmean,
		      int num_data,
		      double *xdata,
		      double *prob)
{
  int i,k;
  double normalization = 1/( (double)num_data*4.0/3.0 );
  for (i=0; i<num_data; i++)
    add_one_point_epan1d(xdata[i],density,win_width/sqrt(prob[i]/gmean));

  density->data_max  =density->map[0]*normalization;
  density->data_min = density->map[0]*normalization;

  for (k=0; k<density->bins; k++)
    {
      density->map[k] = density->map[k]*normalization;
      if (density->map[k]>density->data_max) 
        density->data_max=density->map[k];
      else
        if (density->map[k]< density->data_min)
          density->data_min = density->map[k];
    }

}


		      
double prob_from_map2d(  map2d *map,
                         double x,
                         double y,
                         double xstep,
                         double ystep )       
{
  int xbin=(int)((x - map->x_min)/xstep);
  double xratio=(x-map->x_min-(xstep*xbin))/xstep;
  int ybin=(int)((y-map->y_min)/ystep);
  double yratio=(y-map->y_min-(ystep*ybin))/ystep;
  double dens1,dens2;

  dens1 = map->map[ybin][xbin]+( map->map[ybin][xbin+1]
				 -map->map[ybin][xbin]  )*xratio; 
  dens2 = map->map[ybin+1][xbin]+( map->map[ybin+1][xbin+1]
				   -map->map[ybin+1][xbin]  )*xratio;
  return dens1 + ( dens2 - dens1 ) * yratio;
}

double prob_from_map3d(  map3d *map,
                         double x,
                         double y,
                         double z,
                         double xstep,
                         double ystep,
                         double zstep )       
{
  int xbin=(int)((x - map->x_min)/xstep);
  double xratio=(x-map->x_min-(xstep*xbin))/xstep;
  int ybin=(int)((y-map->y_min)/ystep);
  double yratio=(y-map->y_min-(ystep*ybin))/ystep;
  int zbin=(int)((z-map->z_min)/zstep);
  double zratio=(z-map->z_min-(zstep*zbin))/zstep;

  double dens1, dens2, dens3, dens4;

  dens1 = map->map[zbin][ybin][xbin]+( map->map[zbin][ybin][xbin+1]
				 -map->map[zbin][ybin][xbin]  )*xratio; 
  dens2 = map->map[zbin][ybin+1][xbin]+( map->map[zbin][ybin+1][xbin+1]
				   -map->map[zbin][ybin+1][xbin]  )*xratio;


  dens3 = dens1 + ( dens2 - dens1 ) * yratio;

  dens1 = map->map[zbin+1][ybin][xbin]+( map->map[zbin+1][ybin][xbin+1]
				 -map->map[zbin+1][ybin][xbin]  )*xratio; 
  dens2 = map->map[zbin+1][ybin+1][xbin]+( map->map[zbin+1][ybin+1][xbin+1]
				   -map->map[zbin+1][ybin+1][xbin]  )*xratio;


  dens4 = dens1 + ( dens2 - dens1 ) * yratio;

  return dens3 + ( dens4 - dens3 ) * zratio;
}



double comp_data_probs_2d ( map2d  *density,
                            double xwidth,
                            double ywidth,
                            int    num_data,
			    double *xdata,
			    double *ydata,
			    double *prob)
{
  int i,k,l;
  double normalization = 1/( (double)num_data*xwidth*ywidth*PI/2.0 ), 
         gmean = 0.0;
  double xstep = (density->x_max-density->x_min)/(double)(density->x_bins - 1);
  double ystep = (density->y_max-density->y_min)/(double)(density->y_bins - 1);

  float *p = NULL;
//	printf(" nnnoooo way\n");
//  printf("%f %f %d!\n", density->x_min, density->x_max, num_data);
  for (i=0; i<num_data; i++) {
	  if( (xdata[i] >= density->x_min) && (xdata[i] <= density->x_max) &&
		  (ydata[i] >= density->y_min) && (ydata[i] <= density->y_max)
		  )
		add_one_point_epan(xdata[i],ydata[i],density,xwidth,ywidth);
	  else
		  num_data--;
  }
//  printf("%f %f %d\n", density->y_min, density->y_max, num_data);
  density->data_max  =density->map[0][0]*normalization;
  density->data_min = density->map[0][0]*normalization;
  for (k=0; k<density->y_bins; k++)
    for (l=0; l<density->x_bins; l++)
      {
	density->map[k][l] = density->map[k][l]*normalization;
	if (density->map[k][l]>density->data_max) 
	  density->data_max=density->map[k][l]; // BUG: shouldn't this have a *normalization like above
	else
	  if (density->map[k][l]< density->data_min)
	    density->data_min = density->map[k][l]; // BUG: shouldn't this have a *normalization like above
      }

  for (i=0; i<num_data; i++)
    {
	  if( (xdata[i] >= density->x_min) && (xdata[i] <= density->x_max) &&
		  (ydata[i] >= density->y_min) && (ydata[i] <= density->y_max)
		 ) {
		prob[i]=prob_from_map2d(density,xdata[i],ydata[i],xstep,ystep);
		if(prob[i] != 0)
			gmean += log(prob[i]);
		//else
			//printf("%e %d %e %e %e %e\n", prob[i], i, xdata[i],ydata[i], xstep, ystep);
			
	  }
    }

  for (k=0; k<density->y_bins; k++)
    for (l=0; l<density->x_bins; l++)
      density->map[k][l]=0;

  return exp(gmean/num_data);
}


void map3d_normalize( map3d *density, double normalization ){
  int x,y,z;

  density->data_max  =density->map[0][0][0]*normalization;
  density->data_min = density->map[0][0][0]*normalization;
  for (z=0; z<density->z_bins; z++)
    for (y=0; y<density->y_bins; y++)
      for (x=0; x<density->x_bins; x++){
	density->map[z][y][x] = density->map[z][y][x]*normalization;
	if (density->map[z][y][x]>density->data_max) 
	  density->data_max=density->map[z][y][x];
	else
	  if (density->map[z][y][x]< density->data_min)
	    density->data_min = density->map[z][y][x];
      }

}



double comp_data_probs_3d ( map3d  *density,
                            double xwidth,
                            double ywidth,
                            double zwidth,
                            int    num_data,
			    double *xdata,
			    double *ydata,
			    double *zdata,
			    double *prob)
{
  int i,x,y,z;
  double normalization = 15.0/( (double)num_data*xwidth*ywidth*zwidth*PI*8.0), 
         gmean = 0.0;
  double xstep = (density->x_max-density->x_min)/(double)(density->x_bins - 1);
  double ystep = (density->y_max-density->y_min)/(double)(density->y_bins - 1);
  double zstep = (density->z_max-density->z_min)/(double)(density->z_bins - 1);



  for (i=0; i<num_data; i++){
		if( (xdata[i] >= density->x_min) && (xdata[i] <= density->x_max) &&
		  (ydata[i] >= density->y_min) && (ydata[i] <= density->y_max) &&
		  (zdata[i] >= density->z_min) && (zdata[i] <= density->z_max)
		  )
			add_one_point_epan_3d(xdata[i],ydata[i],zdata[i],
				density,xwidth,ywidth,zwidth);
		else
			num_data--;
		
  }

  map3d_normalize( density, normalization );

  for (i=0; i<num_data; i++)
    {
		if( (xdata[i] >= density->x_min) && (xdata[i] <= density->x_max) &&
		  (ydata[i] >= density->y_min) && (ydata[i] <= density->y_max) &&
		  (zdata[i] >= density->z_min) && (zdata[i] <= density->z_max)
		  )
			prob[i]=prob_from_map3d(density,
			      xdata[i],ydata[i],zdata[i],
			      xstep,ystep,zstep);
		if(prob[i] != 0)
      gmean += log(prob[i]);
    }

  printf("gmean = %e \n",exp(gmean/num_data));

  for (z=0; z<density->z_bins; z++)
    for (y=0; y<density->y_bins; y++)
      for (x=0; x<density->x_bins; x++)
	density->map[z][y][x]=0;

  return exp(gmean/num_data);
}



void comp_density_2d ( map2d  *density,
		       double xwidth,
		       double ywidth,
		       double gmean,
		       int    num_data,
		       double *xdata,
		       double *ydata,
		       double *prob)
{
  int i,k,l;
  double normalization = 1/( (double)num_data*PI/2.0 );

  for (i=0; i<num_data; i++)
	if( (xdata[i] >= density->x_min) && (xdata[i] <= density->x_max) &&
		  (ydata[i] >= density->y_min) && (ydata[i] <= density->y_max)
		  )
		add_one_point_epan2(xdata[i],ydata[i],
			density,
			xwidth/sqrt(prob[i]/gmean),
			ywidth/sqrt(prob[i]/gmean));
  density->data_max  =density->map[0][0]*normalization;
  density->data_min = density->map[0][0]*normalization;
  for (k=0; k<density->y_bins; k++)
    for (l=0; l<density->x_bins; l++)
      {
	density->map[k][l] = density->map[k][l]*normalization;
	if (density->map[k][l]>density->data_max) 
	  density->data_max=density->map[k][l];
	else
	  if (density->map[k][l]< density->data_min)
	    density->data_min = density->map[k][l];
      }
}

void comp_density_3d ( map3d  *density,
		       double xwidth,
		       double ywidth,
		       double zwidth,
		       double gmean,
		       int    num_data,
		       double *xdata,
		       double *ydata,
		       double *zdata,
		       double *prob)
{
  int i;
  double normalization = 15.0/( (double)num_data*PI*8.0 );

  for (i=0; i<num_data; i++)
    add_one_point_epan2_3d(xdata[i],ydata[i],zdata[i],
			   density,
			   xwidth/sqrt(prob[i]/gmean),
			   ywidth/sqrt(prob[i]/gmean),
			   zwidth/sqrt(prob[i]/gmean)
			   );

  map3d_normalize( density, normalization );

}

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




void AVSdens3dwrite(int scaling,
		    map3d *map,
		    char *fname){
  FILE *outfile;
  int x,y,z,i;
  unsigned short *buf = malloc(map->x_bins*map->y_bins*map->z_bins*sizeof(unsigned short));
  avs_header avs_head;

  avs_head.ndim = 3;
  avs_head.dim1 = map->x_bins;
  avs_head.dim2 = map->y_bins;
  avs_head.dim3 = map->y_bins;
  avs_head.min_x = map->x_min;
  avs_head.min_y = map->y_min;
  avs_head.min_z = map->y_min;
  avs_head.max_x = map->x_max;
  avs_head.max_y = map->y_max;
  avs_head.max_z = map->z_max;
  avs_head.filetype = 0;
  avs_head.skip = 0;
  avs_head.nspace = 3;
  avs_head.veclen = 1;
  avs_head.dataname[0] = '\0';
  avs_head.datatype = 2;

  outfile = fopen(fname, "wb");
  if (outfile==NULL) {
    fprintf (stderr, "Error: Can't write the image: %s !", fname);
    return;
  }

  avs_write_header(outfile, &avs_head);
  i=0;
  for (z=0; z<map->z_bins; z++)
    for (y=0; y<map->y_bins; y++)
      for (x=0; x<map->x_bins; x++, i++){
	buf[i]=(unsigned short) (((double)scaling*(map->map[z][y][x]))/map->data_max);
      }
   fwrite(buf, sizeof(unsigned short), (size_t)(map->x_bins*map->y_bins*map->z_bins), outfile);
   fclose(outfile); 
}


int option_do_help = 0;
int option_do_bounds = 0;
int program_flag_debug = 0;
int program_flag_bounds = 0;


int read_line_to_doubles(FILE* input, int line_number, char* line_buf, int max_line_length, double* values, int max_values, int *columns) {
	if(fgets(line_buf, max_line_length, input) == NULL) {
		return 1; // we are done, EOF
	} else {
		if(line_buf[strlen(line_buf)-1] != '\n') {
			fprintf(stderr, "line buffer is too small, increase it (needs recompilation)\n");
			exit(1);
		} else {
			line_buf[strlen(line_buf)-1] = 0;
			if(program_flag_debug)
				printf("got the line [%s]\n", line_buf);
			double value;
			int length = 0;
			int index = 0;
			while(sscanf(line_buf, "%lf %n", &value, &length) >= 1) { // %n shouldn't count as an element, but sometimes does (bug?), so >=1 will do
				if(program_flag_debug)
					printf("value = %lf %d\n", value);
				line_buf += length; // just move the pointer forwards so we skip the current value
				if(index == max_values) {
					fprintf(stderr, "number of columns (%d) bigger than maximum (%d), increase it (needs recompilation)\n", index, max_values);
					exit(2);
				}
				values[index] = value;
				index++;
				if(program_flag_debug)
					printf("line left [%s]\n", line_buf);
			}
			*columns = index;
			if(program_flag_debug)
				printf("done with line\n", value);
			return 0;
		}
	}
}



struct option options[] = {
	{"help", no_argument, 0, 'h'},
	{"bounds", no_argument, 0, 'b'},
	{"debug", no_argument, 0, 'd'},
	{0},
	{"grid", no_argument, 0, 'g'},
	{"initialize", no_argument, 0, 'i'},
	{"optimize", no_argument, 0, 't'},
	{"dimension", required_argument, 0, 'd'},
	{"limit-level-difference", required_argument, 0, 'l'},
	{"subdivides", required_argument, 0, 's'},
	{"output", required_argument, 0, 'o'},
	{"input", optional_argument, 0, 'n'},
	{0}
};

int main__(int argc, char* const *argv) {
	int c;
	int indexptr;
	int max_dim = 100;
	int max_line_length = 10000;
	char* line_buf;
	
	FILE* input;
	
	input = stdin; // by default read from stdin
	
	while((c = getopt_long(argc, argv, "hbd", &options[0], &indexptr)) != -1) {
		switch(c) {
			case 'h':
				printf("usage: %s -d <dim> [options]\n", argv[0]);
				break;
			case 'b':
				program_flag_bounds = 1;
				break;
			case 'd':
				program_flag_debug = 1;
				break;
				
		}
	}

	double* row = calloc(max_dim, sizeof(double));
	line_buf = calloc(max_line_length, sizeof(char));
	int line_number = 1;
	int columns = 0;
	int row_elements = 0;
	if(program_flag_bounds) {
		double* max_values = calloc(max_dim, sizeof(double));
		double* min_values = calloc(max_dim, sizeof(double));
		while( read_line_to_doubles(input, line_number, line_buf, max_line_length, row, max_dim, &row_elements) == 0) {
			if(line_number == 1) {
				columns = row_elements;
				for(int i = 0; i < row_elements; i++) {
					max_values[i] = row[i];
					min_values[i] = row[i];
				}
			} else {
				if(row_elements != columns) {
					fprintf(stderr, "data doesn't make sense, expected %d elements in row/line %d, got %d", columns, line_number, row_elements);
					exit(3);
				}
				for(int i = 0; i < row_elements; i++) {
					if(row[i] > max_values[i])
						max_values[i] = row[i];
					if(row[i] < min_values[i])
						min_values[i] = row[i];
				}
			}
			line_number++;
		}
		if(program_flag_debug) {
			printf("read %d number of lines\n", line_number-1);
		}
		printf("min");
		for(int i = 0; i < row_elements; i++) {
			printf(" %g", min_values[i]);
		}
		printf("\n");
		printf("max");
		for(int i = 0; i < row_elements; i++) {
			printf(" %g", max_values[i]);
		}
		printf("\n");
		//fscanf(input, " %lf",data[j]+i);
		//fscanf(infile," \n");
		//line_number++;
		
	}
}

