#pragma once


typedef unsigned char ubyte;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct ImageGray ImageGray;

struct ImageGray
{
   ulong Width;
   ulong Height;
   ubyte *Pixmap;
};
struct MaxTree;
typedef struct MaxTree MaxTree;

typedef struct DecisionStruct DecisionStruct;

struct DecisionStruct
{
   char *Name;
   void (*Filter)(MaxTree *, ImageGray *, ImageGray *, ImageGray *, double (*attribute)(void *), double);
};

#define NUMDECISIONS 4
extern DecisionStruct Decisions[NUMDECISIONS];


double MaxTreeforSpacesNoFile(ImageGray *img, int CONNECTIVITY,int *nLeaves);
ImageGray *ImageGrayCreate(ulong width, ulong height);
ImageGray *GetTemplate(char *templatefname, ImageGray *img);
void ImageGrayDelete(ImageGray *img);
MaxTree *MaxTreeCreate(ImageGray *img, ImageGray *,
                       void *(*newauxdata)(ulong, ulong, int, ulong *, ImageGray *),
                       void (*addtoauxdata)(void *, ulong, ulong, int, ulong *, ImageGray *),
                       void (*mergeauxdata)(void *, void *),
                       void (*postauxdata)(void *, ubyte),
                       void (*deleteauxdata)(void *),int CONNECTIVITY);
void MaxTreeDelete(MaxTree *mt);
int NumberOfLeaves(MaxTree *mt);
double CalcDynamic(MaxTree *mt,int NumberOfLeaves,double *dyn,ulong *AreaOfInfluence,double *ElongationOfInfluenceZone,double *X,double *Y,double (*attribute)(void *));



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

extern AttribStruct Attribs[NUMATTR];



