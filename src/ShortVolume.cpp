#include "ShortVolume.h"
#include <stdlib.h>
#include <string>

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
ShortVolume::ShortVolume() : SimpleShortVolume() {
  _initVars();
}
ShortVolume::ShortVolume(const ShortVolume& s) : SimpleShortVolume() {
  _initVars();
  copyIntoThis(s);
}
ShortVolume::ShortVolume(const unsigned long& x,
			 const unsigned long& y,
			 const unsigned long& z) : SimpleShortVolume(x,y,z) {
  _initVars();
}

ShortVolume::ShortVolume(const char name[]) : SimpleShortVolume(){
  _initVars();

  string filename(name);

  if (filename.length() < 5) {
    cerr << "Check your filename: " << filename << endl;
    return;
  }

  string file_ext = filename.substr(filename.length()-3, 3);

 /* if (file_ext == "sff")
    readSFF(name);*/
  if (file_ext == "fld")
    readAVS(name);
  else {
    cerr << "File type not supported" << endl;
    exit(1);
  }
}
/*
 * Destructors
 */
ShortVolume::~ShortVolume(){
  _clearHistogram();
 _freeArray();
}

/*
 * Public methoden
 */
void ShortVolume::copyIntoThis(const ShortVolume& s){
  SimpleShortVolume::copyIntoThis(s);
  _maxPosition=s.getMaxPosition();
  _minPosition=s.getMinPosition();
  _maxValue=s.getMaxValue();
  _minValue=s.getMinValue();
  createHistogram();
}  

void ShortVolume::putVoxel(const unsigned long& x,
			   const unsigned long& y,
			   const unsigned long& z,
			   const short& value){
  short oldValue = getVoxel(x,y,z);
  bool newExtrema = false;
  if (value>_maxValue) {
    _maxValue = value;
    _maxPosition.x = x;
    _maxPosition.y = y;
    _maxPosition.z = z;
  } else {
    if ((_maxPosition.x == x) &&
	(_maxPosition.y == y) &&
	(_maxPosition.z == z)) {
      newExtrema = true;
    }
  }
  if (value<_minValue) {
    _minValue=value;
    _minPosition.x = x;
    _minPosition.y = y;
    _minPosition.z = z;
  } else {
    if ((_minPosition.x == x) &&
	(_minPosition.y == y) &&
	(_minPosition.z == z)) {
      newExtrema = true;
    }
  }
  if (newExtrema) _updateExtrema();
  if (hasHistogram()) {
    if ((value >= _histogramMin) && (value <=_histogramMax)) {
      _histogram[oldValue-_histogramMin]--;
      _histogram[value-_histogramMin]++;
    } else {
      //Histogram past niet meer in beschikbare ruimte, dus haal maar weg.
      _clearHistogram();
    }
  }
  SimpleShortVolume::putVoxel(x,y,z,value);
}

bool ShortVolume::hasHistogram() const{
  return (_histogram!=NULL);
}
  
short ShortVolume::getMaxValue() const {
  return _maxValue;
}
short ShortVolume::getMinValue() const {
  return _minValue;
}


voxelPos ShortVolume::getMaxPosition() const {
  return _maxPosition;
}
voxelPos ShortVolume::getMinPosition() const {
  return _minPosition;
}
void ShortVolume::clear(const short& value){
  SimpleShortVolume::clear(value);
  if (hasData()) {
    if (hasHistogram()){
      if ((value >= _histogramMin) && (value <=_histogramMax)) {
	unsigned long range = getXSize()*getYSize()*getZSize();
	for (short i=_histogramMin;i<=_histogramMax;i++){
	  _histogram[i-_histogramMin]=(i==value?range:0);
	}
      } else {
	_clearHistogram();
      }
    }
    _maxValue=value;
    _minValue=value;
    _minPosition.x=0;
    _minPosition.y=0;
    _minPosition.z=0;
    _maxPosition=_minPosition;
  }
} 

void ShortVolume::createHistogram(){
  if (!hasHistogram() && hasData()) {
    _histogramMax = _maxValue;
    _histogramMin = _minValue;
    _histogram = new unsigned long[_histogramMax - _histogramMin + 1];
    for (short i=_histogramMin;i<=_histogramMax;i++){
      _histogram[i-_histogramMin]=0;
    }
    for (unsigned long k=0 ; k<getZSize() ; k++){
      for (unsigned long j=0 ; j<getYSize() ; j++){
	for (unsigned long i=0 ; i<getXSize() ; i++){
	  _histogram[getVoxel(i,j,k)-_histogramMin]++;
	}
      }
    }
  }
}

unsigned long ShortVolume::getNumberOfVoxelsWithValue(short value){
  if ((value > _maxValue) || (value < _minValue) || !hasData()){
    return 0;
  } else {
    if (!hasHistogram()){
      createHistogram();
    }
    return _histogram[value-_minValue];
  }
}

unsigned long ShortVolume::getNumberOfVoxelsWithValue(short value) const{
  if ((value > _maxValue) || (value < _minValue) || !hasData()){
    return 0;
  } else {
    if (!hasHistogram()){
      cout << "No histogram present, bailing out";
     exit(-1);
    }
    return _histogram[value-_minValue];
  }
}
/* 
 * private methoden
 */
void ShortVolume::_initVars(){
  _histogram = NULL;
  _maxValue = MINSHORT;
  _minValue = MAXSHORT;
  _maxPosition.x = 0;
  _maxPosition.y = 0;
  _maxPosition.z = 0;
  _minPosition.x = 0;
  _minPosition.y = 0;
  _minPosition.z = 0;
}

void ShortVolume::_putVoxelQuick(const unsigned long& x,
				 const unsigned long& y,
				 const unsigned long& z,
				 const short& value){
  if (value>_maxValue) {
    _maxValue=value;
    _maxPosition.x = x;
    _maxPosition.y = y;
    _maxPosition.z = z;
  }
  if (value<_minValue) {
    _minValue=value;
    _minPosition.x = x;
    _minPosition.y = y;
    _minPosition.z = z;
  }
  SimpleShortVolume::_putVoxelQuick(x,y,z,value);
}

void ShortVolume::invertVolume(){
  short min,max;
  min = _data[0];
  max = min;
  
  getMinMax(min,max);
  _maxValue=min;
  _minValue=max;
  for (unsigned long k=0 ; k<getZSize() ; k++){
    for (unsigned long j=0 ; j<getYSize() ; j++){
      for (unsigned long i=0 ; i<getXSize() ; i++){
	  _putVoxelQuick(i,j,k,max - getVoxel(i,j,k)+min);
	}
      }
    }
 
}


void ShortVolume::_updateExtrema(){
  if (hasData()){
    _maxValue=getVoxel(0,0,0);
    _maxPosition.x=0;
    _maxPosition.y=0;
    _maxPosition.z=0;
    _minValue=getVoxel(0,0,0);
    _maxPosition = _minPosition;
    short current=0;
    for (unsigned long k=0 ; k<getZSize() ; k++){
      for (unsigned long j=0 ; j<getYSize() ; j++){
	for (unsigned long i=0 ; i<getXSize() ; i++){
	  current = getVoxel(i,j,k);
	  if (current > _maxValue) {
	    _maxValue=current;
	    _maxPosition.x=i;
	    _maxPosition.y=j;
	    _maxPosition.z=k;
	  }
	  if (current < _minValue) {
	    _minValue=current;
	    _minPosition.x=i;
	    _minPosition.y=j;
	    _minPosition.z=k;
	  }
	}
      }
    }
  }
}

void ShortVolume::_clearHistogram(){
  if (hasHistogram()) {
    delete[] _histogram;
    _histogram = NULL;
  }
}


#if 0
void ShortVolume::printCumulativeLaplacianHistogram()
{

  int *hist = new int[_maxValue+1];
  int *hist_area = new int[_maxValue+1];

  for (int i=0; i<_maxValue+1; i++) {
    hist[i] = 0;
    hist_area[i] = 0;
  }

  int laplacian;
  short px, nx, py, ny, pz, nz, c;

  for (int z=1; z<getZSize()-1; z++)
    for (int y=1; y<getYSize()-1; y++)
      for (int x=1; x<getXSize()-1; x++) {
	c = getVoxel(x,y,z);
	px = getVoxel(x-1,y,z);
	nx = getVoxel(x+1,y,z);
	py = getVoxel(x,y-1,z);
	ny = getVoxel(x,y+1,z);
	pz = getVoxel(x,y,z-1);
	nz = getVoxel(x,y,z+1);

	laplacian = 6*c - px - nx - py - ny - pz - nz; 
	hist[c] += laplacian;

	hist_area[c] += (c > px) ? 1 : ((c < px) ? -1 : 0);
	hist_area[c] += (c > nx) ? 1 : ((c < nx) ? -1 : 0);
	hist_area[c] += (c > py) ? 1 : ((c < py) ? -1 : 0);
	hist_area[c] += (c > ny) ? 1 : ((c < ny) ? -1 : 0);
	hist_area[c] += (c > pz) ? 1 : ((c < pz) ? -1 : 0);
	hist_area[c] += (c > nz) ? 1 : ((c < nz) ? -1 : 0);
      }

  for (int i=_maxValue-1; i>=0; i--) {
    hist[i] = hist[i] + hist[i+1];
    hist_area[i] = hist_area[i] + hist_area[i+1];
  }

  for (int i=0; i<=_maxValue; i++)
    cout << i << "\t" << hist[i] << "\t" << hist_area[i] << endl;

  delete[] hist;
  delete[] hist_area;

}

#else


void ShortVolume::printCumulativeLaplacianHistogram()
{

  int *hist = new int[_maxValue+1];
  int *hist_area = new int[_maxValue+1];

  for (int i=0; i<_maxValue+1; i++) {
    hist[i] = 0;
    hist_area[i] = 0;
  }

  int laplacian, k;
  short nb[27], c;   // nb[13] is center
  short min, max;
  int voxelcounter = 0;

  for (unsigned int z=1; z<getZSize()-1; z++)
    for (unsigned int y=1; y<getYSize()-1; y++)
      for (unsigned int x=1; x<getXSize()-1; x++) {

	k = 0;
	min = 0;
	max = 0;
	for (unsigned int zz = z; zz <= z+1; zz++)
	  for (unsigned int yy = y; yy <= y+1; yy++)
	    for (unsigned int xx = x; xx <= x+1; xx++) {
	      c = getVoxel(xx, yy, zz);
	      min = (min > c) ? c : min;
	      max = (max < c) ? c : max;
	    }

	if ( min < max ) {
	  voxelcounter++;
	  for (unsigned int zz = z-1; zz <= z+1; zz++)
	    for (unsigned int yy = y-1; yy <= y+1; yy++)
	      for (unsigned int xx = x-1; xx <= x+1; xx++) {
		nb[k] = getVoxel(xx, yy, zz);
		k++;
	      }
	  c = nb[13];
	  laplacian = 27 * c;
	  for (k = 0; k < 27; k++)
	    laplacian -= nb[k];
	  
	  
	  hist[c] += laplacian;
	  
	  for (k = 0; k < 27; k++)
	    hist_area[c] += (c > nb[k]) ? 1 : ((c < nb[k]) ? -1 : 0);
	}
      }
  for (int i=_maxValue-1; i>=0; i--) {
    hist[i] = hist[i] + hist[i+1];
    hist_area[i] = hist_area[i] + hist_area[i+1];
  }

  for (int i=0; i<=_maxValue; i++)
    cout << i << "\t" << hist[i] << "\t" << hist_area[i] << endl;


  cout << "Active edges: " << voxelcounter << endl;

  delete[] hist;
  delete[] hist_area;

}

#endif
