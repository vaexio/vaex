#pragma once
#include "kerneldensity.h"
#include "maxtreeForSpaces.h"

int ImagePGMBinWrite1D_map(map1d *map, char *fname);
int ImagePGMBinWrite_map(map2d *map, char *fname);
int ImagePGMBinWrite(ImageGray *img, char *fname);
