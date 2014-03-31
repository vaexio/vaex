#include <stdlib.h>
#include "ShortVolume.h"
//#include "MW_Exceptions.h"
#include "MaxTree3d.h"


//extern "C" void MTfor3D(char *fname)
int main(int argc, char *argv[])
{
MaxTree3d *mt;
int nrOfLeaveNodes;
double  quality;
char *fname=argv[1];
ShortVolume *s =new ShortVolume(fname);
 
printf("max tree done\n");
//float xRes,yRes,zRes;
 
  s->createHistogram();
//  s->getResolution(xRes,yRes,zRes); 
  //setScales(xRes,yRes,zRes);

//ShortVolume(fname);
mt = new MaxTree3d(*s, 26);

nrOfLeaveNodes = mt->getNumberOfLeaves();
printf("NumberofLeavesNodes:%d\n",nrOfLeaveNodes);
quality=mt->getLeaves();



printf("Quality:%lf\n",quality);
printf("mean Quality:%lf\n",quality/nrOfLeaveNodes);
return 0;
//return quality;
}