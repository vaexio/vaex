#include <stdlib.h>
#include "ShortVolume.h"
//#include "MW_Exceptions.h"
#include "MaxTree3d.h"
#include "SimpleShortVolume.h"

extern "C" double MTfor3D(char *fname,int *nLeaves)
//int main(int argc, char *argv[])
{
MaxTree3d *mt;
int nrOfLeaveNodes;
double  quality;
//char *fname=argv[1];
ShortVolume *s =new ShortVolume(fname);
 
//float xRes,yRes,zRes;
 
  s->createHistogram();
//  s->getResolution(xRes,yRes,zRes); 
  //setScales(xRes,yRes,zRes);

//ShortVolume(fname);
mt = new MaxTree3d(*s, 26);

s->_clearHistogram();
s->_freeArray();

nrOfLeaveNodes = mt->getNumberOfLeaves();
quality=mt->getLeaves();

printf("NumberofLeavesNodes:%d\n",nrOfLeaveNodes);
nLeaves[0]=nrOfLeaveNodes;

mt->_deleteTree();


return (quality/nrOfLeaveNodes) ;
}
