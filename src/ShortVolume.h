#include "definitions.h"

#include "SimpleShortVolume.h"

#ifndef ShortVolume_h
#define ShortVolume_h

using namespace std;


/********************************************************************
  $Author : Gijs de Vries

 *******************************************************************/ 
class ShortVolume : public SimpleShortVolume {
public:
  ShortVolume();
  ShortVolume(const ShortVolume& s);
  ShortVolume(const unsigned long& x,
	      const unsigned long& y,
	      const unsigned long& z);
  ShortVolume(const char name[]);
  virtual ~ShortVolume();
  virtual void copyIntoThis(const ShortVolume& s);
  virtual void putVoxel(const unsigned long& x,
			const unsigned long& y,
			const unsigned long& z, 
			const short& value);
  bool hasHistogram() const;
  short getMaxValue() const;
  short getMinValue() const;
  voxelPos getMaxPosition() const;
  voxelPos getMinPosition() const;
  void clear(const short& value);
  void clear();
  void createHistogram();
  unsigned long getNumberOfVoxelsWithValue(short value);
  unsigned long getNumberOfVoxelsWithValue(short value) const;
  void printCumulativeLaplacianHistogram();
      void invertVolume();
void _clearHistogram();
protected:
  virtual void _initVars();
  virtual void _putVoxelQuick(const unsigned long& x,
			      const unsigned long& y,
			      const unsigned long& z, 
			      const short& value);
  void _updateExtrema();
 // void _clearHistogram();

private:  
  short _maxValue,_minValue;
  voxelPos _maxPosition,_minPosition;
  
  unsigned long * _histogram;
  short _histogramMax;
  short _histogramMin;
};
#endif /* ShortVolume_h */

