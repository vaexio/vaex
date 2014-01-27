/************************************************************************************/
/* avs_io.h  - Header file for avs_io.c                                             */
/* Usage   :                                                                        */
/* Author  :   Georgios K. Ouzounis                                                 */
/* Dep.    :   IWI - University of Groningen                                        */
/* Vs&Date :   vs 1.0 - 2nd Nov. 2004                                               */  
/************************************************************************************/
#ifdef __cplusplus                                                                                     
extern "C" {                                                                                           
#endif

#ifndef _AVS_IO_H_
#define _AVS_IO_H_


typedef struct avs_header
{
   int   ndim;
   int   dim1, dim2, dim3;    /* volume dims */
   float min_x,min_y,min_z;   /* minmum extend  */
   float max_x,max_y,max_z;   /* maximum extend */
   int   datatype;            /* either short or byte - from SFF file */
   int   filetype;            /* 0: binary, 1: ascii */  
   int   skip;                /* not used here */
   int   nspace;
   int   veclen;
   char  dataname[512];       /* file containing data, NULL otherwise */
} avs_header;


int avs_write_header(FILE *fp, avs_header *header);
int avs_read_header(char *fname, avs_header *header);
int _avs_read_header(FILE *fp, avs_header *header);

#endif
#ifdef __cplusplus
}
#endif
