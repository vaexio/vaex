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

double MaxTreeforSpacesNoFile(ImageGray *img, int CONNECTIVITY,int *nLeaves);
ImageGray *ImageGrayCreate(ulong width, ulong height);


