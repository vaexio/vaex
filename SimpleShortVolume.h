#include "definitions.h"
//#include "ProgressReporter.h"
#ifndef SimpleShortVolume_h
#define SimpleShortVolume_h

/********************************************************************
  $Author : Gijs de Vries

  TODO: 

    Maximum and minimum are not yet garanteed when the current exrema
    is overwritten with putVoxel.
 *******************************************************************/ 

using namespace std;


#if 0
struct voxelPos{
  unsigned long x;
  unsigned long y;
  unsigned long z;
};
#else
struct voxelPos{
  unsigned short x;
  unsigned short y;
  unsigned short z;
};
#endif
inline bool sameVoxel(const voxelPos& p, const voxelPos& q){
  return (p.x == q.x)&&(p.y==q.y)&&(p.z==q.z);
}

class SimpleShortVolume {

public:
  SimpleShortVolume();
  SimpleShortVolume(const SimpleShortVolume& s);
  SimpleShortVolume(const unsigned long& x,
		    const unsigned long& y,
		    const unsigned long& z);
  SimpleShortVolume(const unsigned long& x,
		    const unsigned long& y,
		    const unsigned long& z,
		    char name[]);
 // SimpleShortVolume(char name[]);
  virtual ~SimpleShortVolume();
  bool hasData() const ;
  void _freeArray();
  //void readSFF(const char name[], ProgressReporter *rep = 0);
  void readAVS(const char name[]);
  //void readPVM(const char name[], ProgressReporter *rep = 0);
  //void readTIFFstack(char *name[], int count, ProgressReporter *rep = 0);
  /*void readRAW( const char name[], 
                int xdim, int ydim, int zdim, 
                int offset, int datatype, int byteorder,
                ProgressReporter *rep = 0
               ) ;*/

  //void writeSFF(const char name[]) const;
 // void writeSFF(const char name[], ProgressReporter *rep = 0) const;
  //void writeRAW(const char name[]) const;
  //void writeAVS(const char name[]);
  void writeAVS(const char name[]);
  virtual void copyIntoThis(const SimpleShortVolume& s);

  inline short getVoxel(const unsigned long& x,
			const unsigned long& y,
			const unsigned long& z) const {
    return _data[ x + (_xSize *  (y + (_ySize * z )))] ;
  }

  void getNormal(const unsigned long& x,
		 const unsigned long& y,
		 const unsigned long& z,
		 float *normal);
  
  inline short getVoxel(const voxelPos& v) const {
    return getVoxel(v.x,v.y,v.z);
  }
  short getVoxelSave(const unsigned long& x,
		     const unsigned long& y,
		     const unsigned long& z) const;
  short getVoxelSave(const voxelPos& v) const {
    return getVoxelSave(v.x,v.y,v.z);
  }
  virtual void putVoxel(const unsigned long& x,
			const unsigned long& y,
			const unsigned long& z, 
			const short& value);
  void putVoxel(const voxelPos& v, const short& value) {
    putVoxel(v.x,v.y,v.z,value);
  }
  virtual void putVoxelSave(const unsigned long& x,
			    const unsigned long& y,
			    const unsigned long& z, 
			    const short& value);
  void putVoxelSave(const voxelPos& v, const short& value) {
    putVoxelSave(v.x,v.y,v.z,value);
  }
  bool inBounds(const voxelPos& v) const {
    return inBounds(v.x,v.y,v.z);
  }
  bool inBounds(const unsigned long& x,
                const unsigned long& y,
                const unsigned long& z) const;
  unsigned long getXSize() const;
  unsigned long getYSize() const;
  unsigned long getZSize() const;
  void clear(const short& value);
  void clear();
  short * _data;  // Voor extended moet deze bereikbaar zijn
  void getMinMax(short &min, short &max) const;
  void setResolution(float xRes, float yRes, float zRes){
       _xRes = xRes; _yRes = yRes; _zRes = zRes;
  }
  void getResolution(float &xRes, float &yRes, float &zRes) const{
       xRes = _xRes; yRes = _yRes; zRes = _zRes;
  }
  float getXRes() const {return _xRes;}
  float getYRes() const {return _yRes;}
  float getZRes() const {return _zRes;}

protected:
  virtual void _initVars();
  virtual void _putVoxelQuick(const unsigned long& x,
			      const unsigned long& y,
			      const unsigned long& z, 
			      const short& value);
  // for data
 // void _readSlice(int z, char *name);
  void _allocateArray(const unsigned long& x,
		      const unsigned long& y,
		      const unsigned long& z);
  //void _freeArray();

private:
  unsigned long  _xSize,_ySize,_zSize;
  float          _xRes,_yRes,_zRes;
};
#endif /* SimpleShortVolume_h */

