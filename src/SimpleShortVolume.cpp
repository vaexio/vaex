#include <stdlib.h>
#include <string.h>
#include "SimpleShortVolume.h"
//#include <FL/filename.H>

using namespace std;
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <string>

#include <cassert>
#include <cstdio>
#include "avs_io.h"
//#include "MW_Exceptions.h"
//#include "ddsbase.h"
//#undef __GNUC__
//#include <FreeImage.h>

/********************************************************************
  $Author : Gijs de Vries

  TODO: 

  * Maximum and minimum are not yet garanteed when the current exrema
    are overwritten with putVoxel.
  * SFF Comment handlers
 *******************************************************************/ 

/*
 * Constructors
 */
SimpleShortVolume::SimpleShortVolume(){
  _initVars();
}
SimpleShortVolume::SimpleShortVolume(const SimpleShortVolume& s){
  _initVars();
  copyIntoThis(s);
}
SimpleShortVolume::SimpleShortVolume(const unsigned long& x,
				     const unsigned long& y,
				     const unsigned long& z){
  _initVars();
  _allocateArray(x,y,z);
}

SimpleShortVolume::SimpleShortVolume(const unsigned long& x,
				     const unsigned long& y,
				     const unsigned long& z,
				     char name[]){

  unsigned long i,j,k;
  char byteValue;
  short value;
  

  _initVars();
  _allocateArray(x,y,z);

  ifstream file (name, ios::in | ios::binary);
  /*if (!file.is_open()) {
    throw new MW_FileOpenException();
  }*/

  for (k=0 ; k<z ; k++){
    for (j=0 ; j<y ; j++){
      for (i=0 ; i<x ; i++){
	/*
	file.get(byteValue);
	value = ((unsigned char) byteValue) * 256;
	file.get(byteValue);
	value += ((unsigned char) byteValue);
	*/
	file.get(byteValue);
	value = ((unsigned char) byteValue);
	file.get(byteValue);
	value += 256*((unsigned char) byteValue);
	
        _putVoxelQuick(i,j,k,value);
      }
    }
  }  

  file.close();
}

/*
 * Destructors
 */
SimpleShortVolume::~SimpleShortVolume(){
  _freeArray();
}

/*
 * Public methoden
 */
bool SimpleShortVolume::hasData() const {
  return (_data != NULL);
}

void SimpleShortVolume::readAVS(const char name[])
{
  avs_header header;
  //int datamax, datamin;
  FILE *fp;
  unsigned char *buf;
  short *shortbuf;
  fp = fopen(name,"rb");
 /* if(!fp)
   {
    throw new MW_FileOpenException();
   }*/
   
  /* Read AVS Header */
  _avs_read_header(fp, &header);
  
  /* We Only handle 3D datasets */
  if(header.ndim == 2) header.dim3 = 1;
  
  /* Create a volume */
  _allocateArray(header.dim1, header.dim2, header.dim3);
  _xRes = (header.max_x - header.min_x)/(header.dim1-1);
  _yRes = (header.max_y - header.min_y)/(header.dim2-1);
  _zRes = (header.max_z - header.min_z)/(header.dim3-1);
 
  if ((_xRes == 0 ) ||(_yRes == 0 ) ||(_zRes == 0 )){
    _xRes = 1.0;
    _yRes = 1.0;
    _zRes = 1.0;
  }
   
/*  if (rep!=0){
    rep->minimum(0.0);
    rep->maximum((float) header.dim3);
    rep->resetProgress();
  }*/

  short value;
  switch (header.datatype) {
  case 1: /* bytes */
    buf = (unsigned char *)malloc(header.dim1*header.dim2*sizeof(unsigned char));
    for (int k=0 ; k<header.dim3 ; k++){
      int nRead = fread(buf, sizeof(unsigned char), 
			header.dim1*header.dim2, fp);
      if (nRead != header.dim1*header.dim2 ){ 
	_freeArray();
        free(buf);
	//throw new MW_EOFException();
      }
      for (int j=0 ; j<header.dim2 ; j++){
  	    for (int i=0 ; i<header.dim1 ; i++){                    
	      value=((unsigned char) buf[j*header.dim1+i]);
	      _putVoxelQuick(i,j,k,value);
        }
      }
    /*  if (rep!=0) 
        rep->postProgress((float) k+1);*/
    }  
    free(buf);
    //printf("read AVS: DataType: Bytes\n");
    break;    
  case 2: /* shorts */    
  case 3: /* shorts */    
    // assume reading in NATIVE endian format
    shortbuf=(short *)malloc(header.dim1*header.dim2*sizeof(short));
    for (int k=0 ; k<header.dim3 ; k++){
      int nRead = fread(shortbuf, sizeof(short), 
			header.dim1*header.dim2, fp);
      if (nRead != header.dim1*header.dim2 ){ 
	_freeArray();
        free(shortbuf);
	//throw new MW_EOFException();
      }
      for (int j=0 ; j<header.dim2 ; j++){
  	    for (int i=0 ; i<header.dim1 ; i++){

	      _putVoxelQuick(i,j,k,shortbuf[j*header.dim1+i]);
        }
      }
     /* if (rep!=0)
        rep->postProgress((float) k+1);*/
    }  
    free(shortbuf);
    //printf("read AVS: DataType: Shorts\n");
    break;     
  default:
    //printf("read AVS: Wrong datatype\n");
    _freeArray();
    fclose(fp);
  //  throw new MW_FileFormatException();
  }
  fclose(fp); 

 //free(shortbuf);
 //free(buf);
}

void SimpleShortVolume::writeAVS(const char name[])
{

    avs_header avs_head;
    
    avs_head.ndim = 3;
    avs_head.dim1 = _xSize;
    avs_head.dim2 = _ySize;
    avs_head.dim3 = _zSize;
    avs_head.min_x = 0;
    avs_head.min_y = 0;
    avs_head.min_z = 0;
    avs_head.max_x = _xRes*(float)(_xSize - 1);
    avs_head.max_y = _yRes*(float)(_ySize - 1);
    avs_head.max_z = _zRes*(float)(_zSize - 1);
    avs_head.filetype = 0;
    avs_head.skip = 0;
    avs_head.nspace = 3;
    avs_head.veclen = 1;
    avs_head.dataname[0] = '\0';

    short minVal = _data[0];
    short maxVal = minVal;
    for (unsigned int i=0; i<_xSize*_ySize*_zSize; i++) {
	minVal = _data[i] < minVal ? _data[i] : minVal;
	maxVal = _data[i] > maxVal ? _data[i] : maxVal;
    }
    if ((minVal >= 0) && (maxVal <= 255))
	avs_head.datatype = 1;
    else
	avs_head.datatype = 2;

    FILE *fp;

    fp = fopen(name, "wb");
   /* if(!fp)
    {
      throw new MW_FileOpenException();	
    }*/
    
    /*if (rep!=0){
      rep->minimum(0.0);
      rep->maximum((float) avs_head.dim3);
      rep->resetProgress();
    }*/
    switch(avs_head.datatype){
	case 1:
	    unsigned char byteVal;
	    avs_head.datatype = 1;
	    avs_write_header(fp, &avs_head);
	    for (int k=0 ; k<avs_head.dim3 ; k++) {
		  for (int j=0 ; j<avs_head.dim2 ; j++) { 
		    for (int i=0 ; i<avs_head.dim1 ; i++) {
			  byteVal = (unsigned char) getVoxel(i,j,k);
			  fwrite(&byteVal, sizeof(unsigned char), 1, fp);
		    }
		  }
		/*  if (rep!=0)
		    rep->postProgress((float) k+1);*/
	    }
	    break;
	case 2:
	case 3:
	    avs_head.datatype = 3;
	    avs_write_header(fp, &avs_head);
        for (int k = 0; k<avs_head.dim3; k++){
  	      fwrite(_data + k*_xSize*_ySize, sizeof(short), _xSize*_ySize, fp);
  	    /*  if (rep!=0)
 		    rep->postProgress((float) k+1);  */	      
        }
	    break;
	default:
	    printf("writeAVS: Datatype is not supported\n");
	    fclose(fp);
	    return;
    } 
    fclose(fp);
}


void SimpleShortVolume::copyIntoThis(const SimpleShortVolume& s){
  if (s.hasData()){
    unsigned long x = s.getXSize();
    unsigned long y = s.getYSize();
    unsigned long z = s.getZSize();
    _allocateArray(x,y,z);
    for (unsigned long k=0 ; k<z ; k++){
      for (unsigned long j=0 ; j<y ; j++){
        for (unsigned long i=0 ; i<x ; i++){
          _putVoxelQuick(i,j,k,s.getVoxel(i,j,k));
        }
      }
    }  
  }
}


short SimpleShortVolume::getVoxelSave(const unsigned long& x,
				      const unsigned long& y,
				      const unsigned long& z) const {
  if (( x < 0) | (x >= _xSize) |
      ( y < 0) | (y >= _ySize) |
      ( z < 0) | (z >= _zSize)) {
    //index out of bounds
    cout << "Index out of bounds." << endl;
    exit(-1);
  }
  return getVoxel(x,y,z);
}

void SimpleShortVolume::getNormal(const unsigned long& x,
				  const unsigned long& y,
				  const unsigned long& z,
				  float *normal)
{

  bool nx = x + 1 < _xSize;
  bool ny = y + 1 < _ySize;
  bool nz = z + 1 < _zSize;

  bool px = x - 1 >= 0;
  bool py = y - 1 >= 0;
  bool pz = z - 1 >= 0;

  unsigned long offset;
  offset = x + (_xSize *  (y + (_ySize * z )));
  short pxval = (px) ? _data[offset - 1] : _data[offset];
  short nxval = (nx) ? _data[offset + 1] : _data[offset];
  
  short pyval = (py) ? _data[offset - _xSize] : _data[offset];
  short nyval = (ny) ? _data[offset + _xSize] : _data[offset];
  
  short pzval = (pz) ? _data[offset - _xSize*_ySize] : _data[offset];
  short nzval = (nz) ? _data[offset + _xSize*_ySize] : _data[offset];

  normal[0] = nxval - pxval;
  normal[1] = nyval - pyval;
  normal[2] = nzval - pzval;
  
  double length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
  if ( (-1.0e-8 < length) && (length < 1.0e-8))
    length = 1.0;
  for (int i=0; i<3; i++)
    normal[i] /= length;
  
}

void SimpleShortVolume::putVoxel(const unsigned long& x,
				 const unsigned long& y,
				 const unsigned long& z,
				 const short& value){
  _data[ x + (_xSize *  y ) + ( _xSize * _ySize * z )] = value;
}
void SimpleShortVolume::putVoxelSave(const unsigned long& x,
				     const unsigned long& y,
				     const unsigned long& z,
				     const short& value){
  if (( x < 0) | (x >= _xSize) |
      ( y < 0) | (y >= _ySize) |
      ( z < 0) | (z >= _zSize)) {
    //index out of bounds
    cout << "Index out of bounds." << endl;
    exit(-1);
  }
  putVoxel(x,y,z,value);
}
bool SimpleShortVolume::inBounds(const unsigned long& x,
				 const unsigned long& y,
				 const unsigned long& z) const {
  // een -1 in unsigned levert waarschijnlijk iets >_Size op
  return (( x >= 0) && (x < _xSize) &&
          ( y >= 0) && (y < _ySize) &&
          ( z >= 0) && (z < _zSize));
}
unsigned long SimpleShortVolume::getXSize() const {
  return _xSize;
}
unsigned long SimpleShortVolume::getYSize() const {
  return _ySize;
}
unsigned long SimpleShortVolume::getZSize() const {
  return _zSize;
}
void SimpleShortVolume::clear(){
  clear(0);
}
void SimpleShortVolume::clear(const short& value){
  unsigned long range;
  if (hasData()) {
    range = _xSize*_ySize*_zSize;
    for (unsigned long i=0 ; i<range ; i++) {
      _data[i] = value;
    }
  }
} 
/* 
 * protected methoden
 */
void SimpleShortVolume::_initVars(){
  _data = NULL;
  _xSize=_ySize=_zSize=0;
}

void SimpleShortVolume::_putVoxelQuick(const unsigned long& x,
				       const unsigned long& y,
				       const unsigned long& z,
				       const short& value){
  _data[ x + (_xSize *  y ) + ( _xSize * _ySize * z )] = value;
}

void SimpleShortVolume::_allocateArray(const unsigned long& x,
				       const unsigned long& y,
				       const unsigned long& z){
  
  if (hasData()) {_freeArray();}
  
  _xSize=x;
  _ySize=y;
  _zSize=z;
  _data = new short[x*y*z]; 
 // if (!_data) { throw new MW_MemoryAllocException(); }
}

void SimpleShortVolume::_freeArray(){
  if (hasData()) {
    delete [] _data;
    _data = NULL;
  }
}

void SimpleShortVolume::getMinMax(short &min, short &max) const
{
  min = _data[0];
  max = min;
  for (unsigned int i=0; i<_xSize*_ySize*_zSize; i++) {
    min = (_data[i] < min) ? _data[i] : min;
    max = (_data[i] > max) ? _data[i] : max;
  }

}

