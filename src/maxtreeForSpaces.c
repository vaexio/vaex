/* maxtree3d.c
 * first written in July 20, 2005 by Erik R. Urbach and then modified to include the volume attribute by Florence Tushabe
 * Email: floratush@yahoo.com.
 * Max-tree with a single attribute parameter and an optional template
 * Attribute: Power attribute and others (start program without arguments for
 *            a complete list of attributes available)
 *            Attributes here can use gray value information of the pixels used
 * Decision: Min, Direct, Max, Subtractive (default)
 * Input images: raw (P5) and plain (P2) PGM 8-bit gray-scale images
 * Output image: raw (P5) PGM 8-bit gray-scale image
 * Compilation: gcc -ansi -pedantic -Wall -O3 -o maxtree3c maxtree3c.c -lm
 *
 * Related papers:
 * [1] E. J. Breen and R. Jones.
 *     Attribute openings, thinnings and granulometries.
 *     Computer Vision and Image Understanding.
 *     Vol.64, No.3, Pages 377-389, 1996.
 * [2] P. Salembier and A. Oliveras and L. Garrido.
 *     Anti-extensive connected operators for image and sequence processing.
 *     IEEE Transactions on Image Processing,
 *     Vol.7, Pages 555-570, 1998.
 * [3] E. R. Urbach and M. H. F. Wilkinson.
 *     Shape-Only Granulometries and Grey-Scale Shape Filters.
 *     Proceedings of the ISMM2002,
 *     Pages 305-314, 2002.
 * [4] E. R. Urbach and J. B. T. M. Roerdink and M. H. F. Wilkinson.
 *     Connected Rotation-Invariant Size-Shape Granulometries.
 *     Proceedings of the 17th Int. Conf. Pat. Rec.,
 *     Vol.1, Pages 688-691, 2004.
 */

#include "maxtreeForSpaces.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <strings.h>
#include "pgm.h"



#define NUMLEVELS     256
//#define CONNECTIVITY  4
//#define CONNECTIVITY  2
#define PI 3.14159265358979323846

extern void pixel_qsort(double *pix_arr, int npix);

typedef short bool;
#define false 0
#define true  1



#define MIN(a,b)  ((a<=b) ? (a) : (b))
#define MAX(a,b)  ((a>=b) ? (a) : (b))





typedef struct HQueue
{
   ulong *Pixels;
   ulong Head;
   ulong Tail; /* First free place in queue, or -1 when the queue is full */
} HQueue;



typedef struct MaxNode MaxNode;

struct MaxNode
{
   ulong Parent;
   ulong Area;
   void *Attribute;
   ubyte Level;
   ubyte NewLevel;  /* gray level after filtering */
   ulong NumberOfChildren;
   double Dynamic;
};



/* Status stores the information of the pixel status: the pixel can be
 * NotAnalyzed, InTheQueue or assigned to node k at level h. In this
 * last case Status(p)=k. */
#define ST_NotAnalyzed  -1
#define ST_InTheQueue   -2

typedef struct MaxTree MaxTree;

struct MaxTree
{
   long *Status;
   ulong *NumPixelsBelowLevel;
   ulong *NumNodesAtLevel; /* Number of nodes C^k_h at level h */
   MaxNode *Nodes;
   void *(*NewAuxData)(ulong, ulong, int, ulong *, ImageGray *);
   void (*AddToAuxData)(void *, ulong, ulong, int, ulong *, ImageGray *);
   void (*MergeAuxData)(void *, void *);
   void (*PostAuxData)(void *, ubyte);
   void (*DeleteAuxData)(void *);
};



void MaxTreeDelete(MaxTree *mt);



void PostEmptyData(void *attr, ubyte hparent)
{
} /* PostEmptyData */



/****** Typedefs and functions for area attributes ******************************/

typedef struct AreaData
{
   ulong Area;
} AreaData;

void *NewAreaData(ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   AreaData *areadata;

   areadata = malloc(sizeof(AreaData));
   areadata->Area = 1;
   return(areadata);
} /* NewAreaData */

void DeleteAreaData(void *areaattr)
{
   free(areaattr);
} /* DeleteAreaData */

void AddToAreaData(void *areaattr, ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   AreaData *areadata = areaattr;

   areadata->Area ++;
} /* AddToAreaData */

void MergeAreaData(void *areaattr, void *childattr)
{
   AreaData *areadata = areaattr;
   AreaData *childdata = childattr;

   areadata->Area += childdata->Area;
} /* MergeAreaData */

double AreaAttribute(void *areaattr)
{
   AreaData *areadata = areaattr;
   double area;

   area = areadata->Area;
   return(area);
} /* AreaAttribute */



/****** Typedefs and functions for moment of inertia attributes **************************/

typedef struct InertiaData
{
   ulong Area;
   double SumX, SumY, SumX2, SumY2;
} InertiaData;

void *NewInertiaData(ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   InertiaData *inertiadata;

   inertiadata = malloc(sizeof(InertiaData));
   inertiadata->Area = 1;
   inertiadata->SumX = x;
   inertiadata->SumY = y;
   inertiadata->SumX2 = x*x;
   inertiadata->SumY2 = y*y;
   return(inertiadata);
} /* NewInertiaData */

void DeleteInertiaData(void *inertiaattr)
{
   free(inertiaattr);
} /* DeleteInertiaData */

void AddToInertiaData(void *inertiaattr, ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   InertiaData *inertiadata = inertiaattr;

   inertiadata->Area ++;
   inertiadata->SumX += x;
   inertiadata->SumY += y;
   inertiadata->SumX2 += x*x;
   inertiadata->SumY2 += y*y;
} /* AddToInertiaData */

void MergeInertiaData(void *inertiaattr, void *childattr)
{
   InertiaData *inertiadata = inertiaattr;
   InertiaData *childdata = childattr;

   inertiadata->Area += childdata->Area;
   inertiadata->SumX += childdata->SumX;
   inertiadata->SumY += childdata->SumY;
   inertiadata->SumX2 += childdata->SumX2;
   inertiadata->SumY2 += childdata->SumY2;
} /* MergeInertiaData */

double InertiaAttribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double area, inertia;

   area = inertiadata->Area;
   inertia = inertiadata->SumX2 + inertiadata->SumY2 -
             (inertiadata->SumX * inertiadata->SumX +
              inertiadata->SumY * inertiadata->SumY) / area
             + area / 6.0;
   return(inertia);
} /* InertiaAttribute */

double InertiaDivA2Attribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double inertia, area;

   area = (double)(inertiadata->Area);
   inertia = inertiadata->SumX2 + inertiadata->SumY2 -
             (inertiadata->SumX * inertiadata->SumX +
              inertiadata->SumY * inertiadata->SumY) / area
             + area / 6.0;
   return(inertia*2.0*PI/(area*area));
} /* InertiaDivA2Attribute */

double MeanXAttribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double area, sumx;

   area = inertiadata->Area;
   sumx = inertiadata->SumX;
   return(sumx/area);
} /* MeanXAttribute */

double MeanYAttribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double area, sumy;

   area = inertiadata->Area;
   sumy = inertiadata->SumY;
   return(sumy/area);
} /* MeanYAttribute */



/****** Typedefs and functions for Entropy attributes ******************************/

typedef struct EntropyData
{
   ulong Hist[NUMLEVELS];
} EntropyData;

void *NewEntropyData(ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   EntropyData *entropydata;
   ulong lx=x, ly=y, p;
   int i;

   p = ly*(img->Width) + lx;
   entropydata = malloc(sizeof(EntropyData));
   for (i=0; i<NUMLEVELS; i++)  entropydata->Hist[i] = 0;
   entropydata->Hist[img->Pixmap[p]] = 1;
   return(entropydata);
} /* NewEntropyData */

void DeleteEntropyData(void *entropyattr)
{
   free(entropyattr);
} /* DeleteEntropyData */

void AddToEntropyData(void *entropyattr, ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   EntropyData *entropydata = entropyattr;
   ulong lx=x, ly=y, p;

   p = ly*(img->Width) + lx;
   entropydata->Hist[img->Pixmap[p]] ++;
} /* AddToEntropyData */

void MergeEntropyData(void *entropyattr, void *childattr)
{
   EntropyData *entropydata = entropyattr;
   EntropyData *childdata = childattr;
   int i;

   for (i=0; i<NUMLEVELS; i++)  entropydata->Hist[i] += childdata->Hist[i];
} /* MergeEntropyData */

double EntropyAttribute(void *entropyattr)
{
   EntropyData *entropydata = entropyattr;
   double p[NUMLEVELS];
   double num=0.0, entropy = 0.0;
   int i;

   for (i=0; i<NUMLEVELS; i++)  num += entropydata->Hist[i];
   for (i=0; i<NUMLEVELS; i++)  p[i] = (entropydata->Hist[i])/num;
   for (i=0; i<NUMLEVELS; i++)  entropy += p[i] * (log(p[i]+0.00001)/log(2.0));
   return(-entropy);
} /* EntropyAttribute */



/****** Typedefs and functions for Power attributes ******************************/

typedef struct PowerData
{
   ulong Hist[NUMLEVELS];
   ubyte HCurrent;
   ubyte HParent;
} PowerData;

void *NewPowerData(ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   PowerData *powerdata;
   ulong lx=x, ly=y, p;
   int i;

   p = ly*(img->Width) + lx;
   powerdata = malloc(sizeof(PowerData));
   for (i=0; i<NUMLEVELS; i++)  powerdata->Hist[i] = 0;
   powerdata->Hist[img->Pixmap[p]] = 1;
   powerdata->HCurrent = img->Pixmap[p];
   return(powerdata);
} /* NewPowerData */

void DeletePowerData(void *powerattr)
{
   free(powerattr);
} /* DeletePowerData */

void AddToPowerData(void *powerattr, ulong x, ulong y, int numneighbors, ulong *neighbors, ImageGray *img)
{
   PowerData *powerdata = powerattr;
   ulong lx=x, ly=y, p;

   p = ly*(img->Width) + lx;
   powerdata->Hist[img->Pixmap[p]] ++;
} /* AddToPowerData */

void MergePowerData(void *powerattr, void *childattr)
{
   PowerData *powerdata = powerattr;
   PowerData *childdata = childattr;
   int i;

   for (i=0; i<NUMLEVELS; i++)  powerdata->Hist[i] += childdata->Hist[i];
} /* MergePowerData */

void PostPowerData(void *powerattr, ubyte hparent)
{
   PowerData *powerdata = powerattr;

   powerdata->HParent = hparent;
} /* PostPowerData */

double PowerAttribute(void *powerattr)
{
   PowerData *powerdata = powerattr;
   double power = 0.0, dh;
   int h, i;

   h = powerdata->HCurrent;
   for (i=h; i<NUMLEVELS; i++)
   {
      dh = i - (powerdata->HParent);
      /*power += powerdata->Hist[i] * dh * dh;*/
      power += powerdata->Hist[i] * dh;
   }
   return(power);
} /* PowerAttribute */



/****** Image create/read/write functions ******************************/

ImageGray *ImageGrayCreate(ulong width, ulong height)
{
   ImageGray *img;

   img = malloc(sizeof(ImageGray));
   if (img==NULL)  return(NULL);
   img->Width = width;
   img->Height = height;
   img->Pixmap = malloc(width*height);
   if (img->Pixmap==NULL)
   {
      free(img);
      return(NULL);
   }
   return(img);
} /* ImageGrayCreate */



void ImageGrayDelete(ImageGray *img)
{
   free(img->Pixmap);
   free(img);
} /* ImageGrayDelete */



void ImageGrayInit(ImageGray *img, ubyte h)
{
   memset(img->Pixmap, h, (img->Width)*(img->Height));
} /* ImageGrayInit */



ImageGray *ImagePGMAsciiRead(char *fname)
{
   FILE *infile;
   ImageGray *img;
   ulong width, height, i;
   int c;

   infile = fopen(fname, "r");
   if (infile==NULL)  return(NULL);
   fscanf(infile, "P2\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%lu %lu\n255\n", &width, &height);
   img = ImageGrayCreate(width, height);
   if (img==NULL)
   {
      fclose(infile);
      return(NULL);
   }
   for (i=0; i<width*height; i++)
   {
      fscanf(infile, "%d", &c);
      img->Pixmap[i] = c;
   }
   fclose(infile);
   return(img);
} /* ImagePGMAsciiRead */



ImageGray *ImagePGMBinRead(char *fname)
{
   FILE *infile;
   ImageGray *img;
   ulong width, height;
   int c;

   infile = fopen(fname, "rb");
   if (infile==NULL)  return(NULL);
   fscanf(infile, "P5\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%lu %lu\n255\n", &width, &height);
   img = ImageGrayCreate(width, height);
   if (img)  fread(img->Pixmap, 1, width*height, infile);
   fclose(infile);
   return(img);
} /* ImagePGMBinRead */



ImageGray *ImagePGMRead(char *fname)
{
   FILE *infile;
   char id[4];

   infile = fopen(fname, "r");
   if (infile==NULL)  return(NULL);
   fscanf(infile, "%3s", id);
   fclose(infile);
   if (strcmp(id, "P2")==0)  return(ImagePGMAsciiRead(fname));
   else if (strcmp(id, "P5")==0)  return(ImagePGMBinRead(fname));
   else  return(NULL);
} /* ImagePGMRead */



int ImagePGMBinWrite_depr(ImageGray *img, char *fname)
{
   FILE *outfile;

   outfile = fopen(fname, "wb");
   if (outfile==NULL)  return(-1);
   fprintf(outfile, "P5\n%ld %ld\n255\n", img->Width, img->Height);
   fwrite(img->Pixmap, 1, (size_t)((img->Width)*(img->Height)), outfile);
   fclose(outfile);
   return(0);
} /* ImagePGMBinWrite */



/****** Max-tree routines ******************************/

HQueue *HQueueCreate(ulong imgsize, ulong *numpixelsperlevel)
{
   HQueue *hq;
   int i;

   hq = calloc(NUMLEVELS, sizeof(HQueue));
   if (hq==NULL)  return(NULL);
   hq->Pixels = calloc(imgsize, sizeof(ulong));
   if (hq->Pixels==NULL)
   {
      free(hq);
      return(NULL);
   }
   hq->Head = hq->Tail = 0;
   for (i=1; i<NUMLEVELS; i++)
   {
      hq[i].Pixels = hq[i-1].Pixels + numpixelsperlevel[i-1];
      hq[i].Head = hq[i].Tail = 0;
   }
   return(hq);
} /* HQueueCreate */



void HQueueDelete(HQueue *hq)
{
   free(hq->Pixels);
   free(hq);
} /* HQueueDelete */



#define HQueueFirst(hq,h)  (hq[h].Pixels[hq[h].Head++])
#define HQueueAdd(hq,h,p)  hq[h].Pixels[hq[h].Tail++] = p
#define HQueueNotEmpty(hq,h)  (hq[h].Head != hq[h].Tail)



int GetNeighbors(ubyte *shape, ulong imgwidth, ulong imgsize, ulong p,
                 ulong *neighbors)
{
   ulong x;
   int n=0;

   x = p % imgwidth;
   if ((x<(imgwidth-1)) && (shape[p+1]))      neighbors[n++] = p+1;
   if ((p>=imgwidth) && (shape[p-imgwidth]))  neighbors[n++] = p-imgwidth;
   if ((x>0) && (shape[p-1]))                 neighbors[n++] = p-1;
   p += imgwidth;
   if ((p<imgsize) && (shape[p]))             neighbors[n++] = p;
   return(n);
} /* GetNeighbors */



int MaxTreeFlood(MaxTree *mt, HQueue *hq, ulong *numpixelsperlevel,
                 bool *nodeatlevel, ImageGray *img, ubyte *shape, int h,
                 ulong *thisarea,
                 void **thisattr,int CONNECTIVITY)
/* Returns value >=NUMLEVELS if error */
{
   ulong neighbors[CONNECTIVITY];
   ubyte *pixmap;
   void *attr = NULL, *childattr;
   ulong imgwidth, imgsize, p, q, idx, x, y;
   ulong area = *thisarea, childarea;
   MaxNode *node,*nodeP;
   int numneighbors, i;
   int m;

   imgwidth = img->Width;
   imgsize = imgwidth * (img->Height);
   pixmap = img->Pixmap;
   while(HQueueNotEmpty(hq, h))
   {
      area++;
      p = HQueueFirst(hq, h);
      numneighbors = GetNeighbors(shape, imgwidth, imgsize, p, neighbors);
      x = p % imgwidth;
      y = p / imgwidth;
      if (attr)  mt->AddToAuxData(attr, x, y, numneighbors, neighbors, img);
      else
      {
         attr = mt->NewAuxData(x, y, numneighbors, neighbors, img);
         if (attr==NULL)  return(NUMLEVELS);
         if (*thisattr)  mt->MergeAuxData(attr, *thisattr);
      }
      mt->Status[p] = mt->NumNodesAtLevel[h];
      for (i=0; i<numneighbors; i++)
      {
         q = neighbors[i];
         if (mt->Status[q]==ST_NotAnalyzed)
         {
            HQueueAdd(hq, pixmap[q], q);
            mt->Status[q] = ST_InTheQueue;
            nodeatlevel[pixmap[q]] = true;
            if (pixmap[q] > pixmap[p])
            {
               m = pixmap[q];
               childarea = 0;
               childattr = NULL;
               do
               {
                  m = MaxTreeFlood(mt,hq,numpixelsperlevel,nodeatlevel,img,shape,m, &childarea, &childattr,CONNECTIVITY);
                  if (m>=NUMLEVELS)
                  {
                     mt->DeleteAuxData(attr);
                     return(m);
                  }
               } while (m!=h);
               area += childarea;
               mt->MergeAuxData(attr, childattr);
            }
         }
      }
   }
   mt->NumNodesAtLevel[h] = mt->NumNodesAtLevel[h]+1;
   m = h-1;
   while ((m>=0) && (nodeatlevel[m]==false))  m--;
   if (m>=0)
   {
      node = mt->Nodes + (mt->NumPixelsBelowLevel[h] + mt->NumNodesAtLevel[h]-1);
      node->Parent = mt->NumPixelsBelowLevel[m] + mt->NumNodesAtLevel[m];
      mt->PostAuxData(attr, m);
      nodeP= mt->Nodes+mt->NumPixelsBelowLevel[m] + mt->NumNodesAtLevel[m];
      nodeP->NumberOfChildren++;	
   } else {
      idx = mt->NumPixelsBelowLevel[h];
      node = mt->Nodes + idx;
      node->Parent = idx;
      mt->PostAuxData(attr, h);
      node->NumberOfChildren++;
   }
   node->Area = area;
   node->Attribute = attr;
   node->Level = h;
   nodeatlevel[h] = false;
   *thisarea = area;
   *thisattr = attr;
   return(m);
} /* MaxTreeFlood */



MaxTree *MaxTreeCreate(ImageGray *img, ImageGray *template,
                       void *(*newauxdata)(ulong, ulong, int, ulong *, ImageGray *),
                       void (*addtoauxdata)(void *, ulong, ulong, int, ulong *, ImageGray *),
                       void (*mergeauxdata)(void *, void *),
                       void (*postauxdata)(void *, ubyte),
                       void (*deleteauxdata)(void *),int CONNECTIVITY)
{
   ulong numpixelsperlevel[NUMLEVELS];
   bool nodeatlevel[NUMLEVELS];
   HQueue *hq;
   MaxTree *mt;
   ubyte *pixmap = img->Pixmap;
   void *attr = NULL;
   ulong imgsize, p, m=0, area=0;
   int l;

   /* Allocate structures */
   mt = malloc(sizeof(MaxTree));
   if (mt==NULL)  return(NULL);
   imgsize = (img->Width)*(img->Height);
   mt->Status = calloc((size_t)imgsize, sizeof(long));
   if (mt->Status==NULL)
   {
      free(mt);
      return(NULL);
   }
   mt->NumPixelsBelowLevel = calloc(NUMLEVELS, sizeof(ulong));
   if (mt->NumPixelsBelowLevel==NULL)
   {
      free(mt->Status);
      free(mt);
      return(NULL);
   }
   mt->NumNodesAtLevel = calloc(NUMLEVELS, sizeof(ulong));
   if (mt->NumNodesAtLevel==NULL)
   {
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return(NULL);
   }
   mt->Nodes = calloc((size_t)imgsize, sizeof(MaxNode));
   if (mt->Nodes==NULL)
   {
      free(mt->NumNodesAtLevel);
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return(NULL);
   }

   /* Initialize structures */
   for (p=0; p<imgsize; p++)  mt->Status[p] = ST_NotAnalyzed;
   //bzero(nodeatlevel, NUMLEVELS*sizeof(bool));
   //bzero(numpixelsperlevel, NUMLEVELS*sizeof(ulong));
   memset(nodeatlevel, 0, NUMLEVELS*sizeof(bool));
   memset(numpixelsperlevel, 0, NUMLEVELS*sizeof(ulong));
   /* Following bzero is redundant, array is initialized by calloc */
   /* bzero(mt->NumNodesAtLevel, NUMLEVELS*sizeof(ulong)); */
   for (p=0; p<imgsize; p++)  numpixelsperlevel[pixmap[p]]++;
   mt->NumPixelsBelowLevel[0] = 0;
   for (l=1; l<NUMLEVELS; l++)
   {
      mt->NumPixelsBelowLevel[l] = mt->NumPixelsBelowLevel[l-1] + numpixelsperlevel[l-1];
   }
   hq = HQueueCreate(imgsize, numpixelsperlevel);
   if (hq==NULL)
   {
      free(mt->Nodes);
      free(mt->NumNodesAtLevel);
      free(mt->NumPixelsBelowLevel);
      free(mt->Status);
      free(mt);
      return(NULL);
   }

   /* Find pixel m which has the lowest intensity l in the image */
   for (p=0; p<imgsize; p++)
   {
      if (pixmap[p]<pixmap[m])  m = p;
   }
   l = pixmap[m];

   /* Add pixel m to the queue */
   nodeatlevel[l] = true;
   HQueueAdd(hq, l, m);
   mt->Status[m] = ST_InTheQueue;

   /* Build the Max-tree using a flood-fill algorithm */
   mt->NewAuxData = newauxdata;
   mt->AddToAuxData = addtoauxdata;
   mt->MergeAuxData = mergeauxdata;
   mt->PostAuxData = postauxdata;
   mt->DeleteAuxData = deleteauxdata;
   l = MaxTreeFlood(mt, hq, numpixelsperlevel, nodeatlevel, img,
                    template->Pixmap, l, &area, &attr,CONNECTIVITY);
   if (l>=NUMLEVELS)  MaxTreeDelete(mt);
   HQueueDelete(hq);
   return(mt);
} /* MaxTreeCreate */



void MaxTreeDelete(MaxTree *mt)
{
   void *attr;
   ulong i;
   int h;

   for (h=0; h<NUMLEVELS; h++)
   {
      for (i=0; i<mt->NumNodesAtLevel[h]; i++)
      {
         attr = mt->Nodes[mt->NumPixelsBelowLevel[h]+i].Attribute;
         if (attr)  mt->DeleteAuxData(attr);
      }
   }
   free(mt->Nodes);
   free(mt->NumNodesAtLevel);
   free(mt->NumPixelsBelowLevel);
   free(mt->Status);
   free(mt);
} /* MaxTreeDelete */



void MaxTreeFilterMin(MaxTree *mt, ImageGray *img, ImageGray *template,
                      ImageGray *out, double (*attribute)(void *),
                      double lambda)
{
   MaxNode *node, *parnode;
   ubyte *shape = template->Pixmap;
   ulong i, idx, parent;
   int l;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         parent = node->Parent;
	 parnode = &(mt->Nodes[parent]);
	 if ((idx!=parent) && (((*attribute)(node->Attribute)<lambda) || (parnode->Level!=parnode->NewLevel)))
         {
            node->NewLevel = parnode->NewLevel;
         } else  node->NewLevel = node->Level;
      }
   }
   for (i=0; i<(img->Width)*(img->Height); i++)
   {
      if (shape[i])
      {
         idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
         out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
   }
} /* MaxTreeFilterMin */



void MaxTreeFilterDirect(MaxTree *mt, ImageGray *img, ImageGray *template,
                         ImageGray *out, double (*attribute)(void *),
                         double lambda)
{
   MaxNode *node;
   ubyte *shape = template->Pixmap;
   ulong i, idx, parent;
   int l;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         parent = node->Parent;
         if ((idx!=parent) && ((*attribute)(node->Attribute)<lambda))
         {
            node->NewLevel = mt->Nodes[parent].NewLevel;
         } else  node->NewLevel = node->Level;
      }
   }
   for (i=0; i<(img->Width)*(img->Height); i++)
   {
      if (shape[i])
      {
         idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
         out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
   }
} /* MaxTreeFilterDirect */



void MaxTreeFilterMax(MaxTree *mt, ImageGray *img, ImageGray *template,
                      ImageGray *out, double (*attribute)(void *),
                      double lambda)
{
   MaxNode *node;
   ubyte *shape = template->Pixmap;
   ulong i, idx, parent;
   int l;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         parent = node->Parent;
         if ((idx!=parent) && ((*attribute)(node->Attribute)<lambda))
         {
            node->NewLevel = mt->Nodes[parent].NewLevel;
         } else  node->NewLevel = node->Level;
      }
   }
   for (l=NUMLEVELS-1; l>0; l--)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         parent = node->Parent;
         if ((idx!=parent) && (node->NewLevel==node->Level))
         {
            mt->Nodes[parent].NewLevel = mt->Nodes[parent].Level;
         }
      }
   }
   for (i=0; i<(img->Width)*(img->Height); i++)
   {
      if (shape[i])
      {
         idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
         out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
   }
} /* MaxTreeFilterMax */



void MaxTreeFilterSubtractive(MaxTree *mt, ImageGray *img, ImageGray *template,
                              ImageGray *out, double (*attribute)(void *),
                              double lambda)
{
   MaxNode *node, *parnode;
   ubyte *shape = template->Pixmap;
   ulong i, idx, parent;
   int l;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         parent = node->Parent;
         parnode = &(mt->Nodes[parent]);
         if ((idx!=parent) && ((*attribute)(node->Attribute)<lambda))
         {
            node->NewLevel = parnode->NewLevel;
         } else  node->NewLevel = ((int)(node->Level)) + ((int)(parnode->NewLevel)) - ((int)(parnode->Level));
      }
   }
   for (i=0; i<(img->Width)*(img->Height); i++)
   {
      if (shape[i])
      {
         idx = mt->NumPixelsBelowLevel[img->Pixmap[i]] + mt->Status[i];
         out->Pixmap[i] = mt->Nodes[idx].NewLevel;
      }
   }
} /* MaxTreeFilterSubtractive */



void PrintFilterStatistics(MaxTree *mt, double (*attribute)(void *),
                           ImageGray *img, ImageGray *out)
{
   MaxNode *node;
   double v, vmin=-1.0, vmax=-1.0;
   ulong imgsize, p, idx, num=0, numtotal=0, diff=0, power=0, area0=0, area1=0, areamin=0, areamax=0;
   int l, hold, hnew, isfirst=1;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (p=0; p<mt->NumNodesAtLevel[l]; p++)
      {
         idx = mt->NumPixelsBelowLevel[l] + p;
         node = &(mt->Nodes[idx]);
         if (idx != node->Parent)
         {
            numtotal++;
            if (node->NewLevel != node->Level)  num ++;
            v = (*attribute)(node->Attribute);
            if (isfirst)
            {
               vmin = vmax = v;
               area0 = area1 = areamin = areamax = node->Area;
               isfirst = 0;
            } else {
               if (v<vmin)  {vmin = v; area0 = node->Area;}
               if (v>vmax)  {vmax = v; area1 = node->Area;}
               if (node->Area<areamin)  areamin = node->Area;
               if (node->Area>areamax)  areamax = node->Area;
            }
         }
      }
   }
   printf("Area min. : %ld\n", areamin);
   printf("Area max. : %ld\n", areamax);
   printf("Attribute min. value: %f  (area=%ld)\n", vmin, area0);
   printf("Attribute max. value: %f  (area=%ld)\n", vmax, area1);
   printf("#Max-tree nodes deleted: %ld  (%f%%)\n", num, num*100.0/numtotal);

   num = 0;
   imgsize = (img->Width) * (img->Height);
   for (p=0; p<imgsize; p++)
   {
      hold = img->Pixmap[p];
      hnew = out->Pixmap[p];
      if (hold!=hnew)  num++;
      diff += abs(hold-hnew);
      power += (hold-hnew)*(hold-hnew);
   }
   printf("#pixels changed: %ld  (%f%%)\n", num, num*100.0/imgsize);
   printf("Summed absolute difference: %ld\n", diff);
   printf("Summed squared difference: %ld\n", power);
} /* PrintFilterStatistics */



ImageGray *GetTemplate(char *templatefname, ImageGray *img)
{
   ImageGray *template;

   if (templatefname)
   {
      template = ImagePGMRead(templatefname);
      if (template==NULL)  return(NULL);
      if ((img->Width != template->Width) || (img->Height != template->Height))
      {
	 ImageGrayDelete(template);
         return(NULL);
      }
   } else {
      template = ImageGrayCreate(img->Width, img->Height);
      if (template)  ImageGrayInit(template, NUMLEVELS-1);
   }
   return(template);
} /* GetTemplate */


typedef struct AttribStruct AttribStruct;

struct AttribStruct
{
   char *Name;
   void *(*NewAuxData)(ulong, ulong, int, ulong *, ImageGray *);
   void (*DeleteAuxData)(void *);
   void (*AddToAuxData)(void *, ulong, ulong, int, ulong *, ImageGray *);
   void (*MergeAuxData)(void *, void *);
   void (*PostAuxData)(void *, ubyte);
   double (*Attribute)(void *);
};

#define NUMATTR 7

AttribStruct Attribs[NUMATTR] =
{
  {"Area", NewAreaData, DeleteAreaData, AddToAreaData, MergeAreaData, PostEmptyData, AreaAttribute},
  {"Moment of Inertia", NewInertiaData, DeleteInertiaData, AddToInertiaData, MergeInertiaData, PostEmptyData, InertiaAttribute},
  {"Elongation: (Moment of Inertia) / (area)^2", NewInertiaData, DeleteInertiaData, AddToInertiaData, MergeInertiaData, PostEmptyData, InertiaDivA2Attribute},
  {"Mean X position", NewInertiaData, DeleteInertiaData, AddToInertiaData, MergeInertiaData, PostEmptyData, MeanXAttribute},
  {"Mean Y position", NewInertiaData, DeleteInertiaData, AddToInertiaData, MergeInertiaData, PostEmptyData, MeanYAttribute},
  {"Entropy", NewEntropyData, DeleteEntropyData, AddToEntropyData, MergeEntropyData, PostEmptyData, EntropyAttribute},
  {"Volume", NewPowerData, DeletePowerData, AddToPowerData, MergePowerData, PostPowerData, PowerAttribute}
};


typedef struct DecisionStruct DecisionStruct;

struct DecisionStruct
{
   char *Name;
   void (*Filter)(MaxTree *, ImageGray *, ImageGray *, ImageGray *, double (*attribute)(void *), double);
};

#define NUMDECISIONS 4

DecisionStruct Decisions[NUMDECISIONS] =
{
   {"Min", MaxTreeFilterMin},
   {"Direct", MaxTreeFilterDirect},
   {"Max", MaxTreeFilterMax},
   {"Subtractive", MaxTreeFilterSubtractive},
};

int NumberOfLeaves(MaxTree *mt)
{
int l,i,numLeaves=0;
ulong idx;
MaxNode *node;

 for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         if (node->NumberOfChildren==0)
             numLeaves++;

      }
    }
//printf("NumLeaves:%d\n",numLeaves);
return numLeaves;
}

double CalcDynamic(MaxTree *mt,int NumberOfLeaves,double *dyn,ulong *AreaOfInfluence,double *ElongationOfInfluenceZone,double *X,double *Y,double (*attribute)(void *))
{
int k,l,i,d=0;
ulong idx,idx1;
MaxNode *node,*nodeP,*nodePold;
bool fork=false;
double sumD=0,mean,var=0,quality=0.0;

 for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<mt->NumNodesAtLevel[l]; i++)
      {
         idx = mt->NumPixelsBelowLevel[l] + i;
         node = &(mt->Nodes[idx]);
         nodePold=nodeP= &(mt->Nodes[idx]);
         if (node->NumberOfChildren==0)
           {   
		while(!fork)
               {
                
                 idx1=nodeP->Parent;
                nodeP= &(mt->Nodes[idx1]);
		//AreaOfInfluence[d]=AreaOfInfluence[d]+(double)nodeP->Area;
               // printf("area=%d\n",nodeP->Area);
		//area[d]=area[d]+nodeP->Area;
                if (nodeP->NumberOfChildren>1)
		 {
		  AreaOfInfluence[d]=nodePold->Area;
                  ElongationOfInfluenceZone[d]=(*attribute)(nodePold->Attribute);
                  X[d]=(*Attribs[3].Attribute)(nodePold->Attribute);
 		  Y[d]=(*Attribs[4].Attribute)(nodePold->Attribute);	
		  node->Dynamic=((double)(node->Level-nodeP->Level)/(double)(node->Level));
                  dyn[d]=((double)(node->Level-nodeP->Level)/(double)(node->Level));
		 // printf("Dynamic:%lf,Area:%u,Elongation:%lf,%lf,%lf\n",node->Dynamic,AreaOfInfluence[d],ElongationOfInfluenceZone[d],X[d],Y[d]);
                 
                      quality=quality+dyn[d];
                 
		  d++;
                  
                  fork=true;
		 }
                nodePold=nodeP;
                }
	    }
	fork=false;	
       }
    }
if(d<2)
{
quality=0.0;
//printf("Quality:%lf\n",quality);
}
//else
//printf("Quality:%lf\n",quality);


return quality;
}

double MaxTreeforSpaces(char *imname,int CONNECTIVITY,int *nLeaves)
{
   ImageGray *img, *template, *out;
   MaxTree *mt;
   char *imgfname, *templatefname = NULL, *outfname = "out.pgm";
   double lambda=2;
   int attrib=2, decision=3, r,NumLeaves,i;
   double *dyn,*ElongationOfInflunceZone,*Xextent,*Yextent,quality;
   ulong *AreaOfInfluence;

 
   imgfname = imname;
  // attrib = atoi(argv[2]);
   //lambda = atof(argv[3]);
  // if (argc>=5)  decision = atoi(argv[4]);
   //if (argc>=6)  outfname = argv[5];
  // if (argc>=7)  templatefname = argv[6];
   img = ImagePGMRead(imgfname);
   if (img==NULL)
   {
      fprintf(stderr, "Can't read image '%s'\n", imgfname);
      return(-1);
   }
   template = GetTemplate(templatefname, img);
   if (template==NULL)
   {
      fprintf(stderr, "Can't create template\n");
      ImageGrayDelete(img);
      return(-1);
   }
  // printf("Filtering image '%s' using attribute '%s' with lambda=%f\n", imgfname, Attribs[attrib].Name, lambda);
  // printf("Decision rule: %s   Template: ", Decisions[decision].Name);
   //if (templatefname==NULL)//  printf("<not used>\n");
   //else  printf("%s\n", templatefname);
   //printf("Image: Width=%ld Height=%ld\n", img->Width, img->Height);
   out = ImageGrayCreate(img->Width, img->Height);
   if (out==NULL)
   {
      fprintf(stderr, "Can't create output image\n");
      ImageGrayDelete(template);
      ImageGrayDelete(img);
      return(-1);
   }
   mt = MaxTreeCreate(img, template, Attribs[attrib].NewAuxData, Attribs[attrib].AddToAuxData, Attribs[attrib].MergeAuxData, Attribs[attrib].PostAuxData, Attribs[attrib].DeleteAuxData,CONNECTIVITY);
   if (mt==NULL)
   {
      fprintf(stderr, "Can't create Max-tree\n");
      ImageGrayDelete(out);
      ImageGrayDelete(template);
      ImageGrayDelete(img);
      return(-1);
   }
   NumLeaves=NumberOfLeaves(mt);
   nLeaves[0]=NumLeaves;
   dyn = (double *)calloc(NumLeaves,sizeof(double));
   AreaOfInfluence= (ulong *)calloc(NumLeaves,sizeof(ulong));
   ElongationOfInflunceZone= (double *)calloc(NumLeaves,sizeof(double));
   Xextent=(double *)calloc(NumLeaves,sizeof(double));
   Yextent=(double *)calloc(NumLeaves,sizeof(double));
   quality=CalcDynamic(mt,NumLeaves,dyn,AreaOfInfluence,ElongationOfInflunceZone,Xextent,Yextent,Attribs[attrib].Attribute);
   
   FILE *outfile = fopen("InterestingSubspacesNew.txt","a");

  // pixel_qsort (dyn,NumLeaves);
  fprintf(outfile,"NumberOfLocalMaxima:%d\nQuality:%lf\n",NumLeaves,quality);
  fprintf(outfile,"%s\n","Dynamics,AreaOfInfluence,ElongatonOfInfluenceArea,X-extent,Y-extent");
   for (i=NumLeaves-1;i>=0;i--)
      fprintf(outfile,"%lf,%u,%lf,%lf,%lf\n",dyn[i],AreaOfInfluence[i],ElongationOfInflunceZone[i],Xextent[i],Yextent[i]);

   Decisions[decision].Filter(mt, img, template, out, Attribs[attrib].Attribute, lambda);
  //PrintFilterStatistics(mt, Attribs[attrib].Attribute, img, out);
   MaxTreeDelete(mt);
   r = ImagePGMBinWrite(out, outfname);
  // if (r)  fprintf(stderr, "Error writing image '%s'\n", outfname);
   //else  printf("Filtered image written to '%s'\n", outfname);
   fclose(outfile);
   ImageGrayDelete(out);
   ImageGrayDelete(template);
   ImageGrayDelete(img);
   free(AreaOfInfluence);
   free(ElongationOfInflunceZone);
   free(Xextent);
   free(Yextent);
   free(dyn);

   return (quality/NumLeaves);
} /* main */

double MaxTreeforSpacesNoFile(ImageGray *img, int CONNECTIVITY,int *nLeaves)
{
   ImageGray *template, *out;
   MaxTree *mt;
   char *imgfname, *templatefname = NULL, *outfname = "out.pgm";
   double lambda=2;
   int attrib=2, decision=3, r,NumLeaves,i;
   double *dyn,*ElongationOfInflunceZone,*Xextent,*Yextent,quality;
   ulong *AreaOfInfluence;

 
   template = GetTemplate(templatefname, img);
   if (template==NULL)
   {
      fprintf(stderr, "Can't create template\n");
      ImageGrayDelete(img);
      return(-1);
   }
  // printf("Filtering image '%s' using attribute '%s' with lambda=%f\n", imgfname, Attribs[attrib].Name, lambda);
  // printf("Decision rule: %s   Template: ", Decisions[decision].Name);
   //if (templatefname==NULL)//  printf("<not used>\n");
   //else  printf("%s\n", templatefname);
   //printf("Image: Width=%ld Height=%ld\n", img->Width, img->Height);
   out = ImageGrayCreate(img->Width, img->Height);
   if (out==NULL)
   {
      fprintf(stderr, "Can't create output image\n");
      ImageGrayDelete(template);
      ImageGrayDelete(img);
      return(-1);
   }
   mt = MaxTreeCreate(img, template, Attribs[attrib].NewAuxData, Attribs[attrib].AddToAuxData, Attribs[attrib].MergeAuxData, Attribs[attrib].PostAuxData, Attribs[attrib].DeleteAuxData,CONNECTIVITY);
   if (mt==NULL)
   {
      fprintf(stderr, "Can't create Max-tree\n");
      ImageGrayDelete(out);
      ImageGrayDelete(template);
      ImageGrayDelete(img);
      return(-1);
   }
   NumLeaves=NumberOfLeaves(mt);
   nLeaves[0]=NumLeaves;
   dyn = (double *)calloc(NumLeaves,sizeof(double));
   AreaOfInfluence= (ulong *)calloc(NumLeaves,sizeof(ulong));
   ElongationOfInflunceZone= (double *)calloc(NumLeaves,sizeof(double));
   Xextent=(double *)calloc(NumLeaves,sizeof(double));
   Yextent=(double *)calloc(NumLeaves,sizeof(double));
   quality=CalcDynamic(mt,NumLeaves,dyn,AreaOfInfluence,ElongationOfInflunceZone,Xextent,Yextent,Attribs[attrib].Attribute);
   
   FILE *outfile = fopen("InterestingSubspacesNew.txt","a");

  // pixel_qsort (dyn,NumLeaves);
  fprintf(outfile,"NumberOfLocalMaxima:%d\nQuality:%lf\n",NumLeaves,quality);
  fprintf(outfile,"%s\n","Dynamics,AreaOfInfluence,ElongatonOfInfluenceArea,X-extent,Y-extent");
   for (i=NumLeaves-1;i>=0;i--)
      fprintf(outfile,"%lf,%u,%lf,%lf,%lf\n",dyn[i],AreaOfInfluence[i],ElongationOfInflunceZone[i],Xextent[i],Yextent[i]);

   Decisions[decision].Filter(mt, img, template, out, Attribs[attrib].Attribute, lambda);
  //PrintFilterStatistics(mt, Attribs[attrib].Attribute, img, out);
   MaxTreeDelete(mt);
   r = ImagePGMBinWrite(out, outfname);
  // if (r)  fprintf(stderr, "Error writing image '%s'\n", outfname);
   //else  printf("Filtered image written to '%s'\n", outfname);
   fclose(outfile);
   ImageGrayDelete(out);
   ImageGrayDelete(template);
   ImageGrayDelete(img);
   free(AreaOfInfluence);
   free(ElongationOfInflunceZone);
   free(Xextent);
   free(Yextent);
   free(dyn);

   return (quality/NumLeaves);
} /* main */

int main(int argc, char** argv) {
	int numLeaves;
	double quality;
	if(argc == 2) {
		quality = MaxTreeforSpaces(argv[1], 2, &numLeaves);
		printf("quality: %f\n", quality);
		printf("numLeaves: %d\n", numLeaves);
	} else {
		fprintf(stderr, "usage: %s <filename>\n", argv[0]);
		return 1;
	}
}

