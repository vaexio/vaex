#include "pgm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int ImagePGMBinWrite1D_map(map1d *map, char *fname)
{
   FILE *outfile;
   int k,l,i;
   unsigned char *buf = malloc(map->bins*sizeof(unsigned char));

   outfile = fopen(fname, "wb");
   if (outfile==NULL) {
      fprintf (stderr, "Error: Can't write the image: %s !", fname);
      return(0);
   }
   fprintf(outfile, "P5\n%d %d\n255\n", map->bins,1);
 
   i=0;
   //for(l=map->y_bins - 1;l>=0;l--)
     for (k=0;k<map->bins;k++,i++)
       buf[i] = (unsigned char) ((255.0*sqrt(map->map[k]))/sqrt(map->data_max)); 
   fwrite(buf, 1, (size_t)(map->bins), outfile);
 
   fclose(outfile);
   free(buf);
   return(1);
} /* ImagePGMBinWrite */


int ImagePGMBinWrite_map(map2d *map, char *fname)
{
   FILE *outfile;
   int k,l,i;
   unsigned char *buf = malloc(map->x_bins*map->y_bins*sizeof(unsigned char));

   outfile = fopen(fname, "wb");
   if (outfile==NULL) {
      fprintf (stderr, "Error: Can't write the image: %s !", fname);
      return(0);
   }
   fprintf(outfile, "P5\n%d %d\n255\n", map->x_bins, map->y_bins);
 
   i=0;
   for(l=map->y_bins - 1;l>=0;l--)
     for (k=0;k<map->x_bins;k++,i++)
       buf[i] = (unsigned char) ((255.0*sqrt(map->map[l][k]))/sqrt(map->data_max)); 
   fwrite(buf, 1, (size_t)(map->x_bins*map->y_bins), outfile);
 
   fclose(outfile);
   return(1);
} /* ImagePGMBinWrite */


int ImagePGMBinWrite(ImageGray *img, char *fname)
{
   FILE *outfile;

   outfile = fopen(fname, "wb");
   if (outfile==NULL)  return(-1);
   fprintf(outfile, "P5\n%ld %ld\n255\n", img->Width, img->Height);
   fwrite(img->Pixmap, 1, (size_t)((img->Width)*(img->Height)), outfile);
   fclose(outfile);
   return(0);
} /* ImagePGMBinWrite */
