#include "definitions.h"

#include "ShortVolume.h"
#include "MTAuxData.h"
#include "math.h"

#include "Queue.h"
//#include "Abstr_MBox.h"
//#include "ProgressReporter.h"
#ifndef MaxTree3D_h
#define MaxTree3D_h

using namespace std;

struct maxTreeNode {
  // node vars
  double attribute;
  // voxelvars
  unsigned short level;
  unsigned short currentValue;
  unsigned short peakLevel;    // additions for k-flat filtering
  // voxellijst
  unsigned long voxelOffset;
  unsigned long numberOfVoxels;
  // stamboom gegevens
  maxTreeNode * parent;
  unsigned long numberOfChildren;
  maxTreeNode ** children;
  // edge list for fast isosurfacing
  unsigned long numberOfEdges;
  unsigned long *edgeList;
  bool processed;
  MTAuxData *auxdata;
  double Dynamic;
};



class MaxTree3d {

public:
  MaxTree3d();
  MaxTree3d(const MaxTree3d& mt);
  MaxTree3d(const ShortVolume& s, MTAuxData * mta, int con);
  MaxTree3d(const ShortVolume& s, int con);
  MaxTree3d(const ShortVolume& s, const ShortVolume& mask, MTAuxData * mta, int con);
  MaxTree3d(const ShortVolume& s, const ShortVolume& mask, int con);

  ~MaxTree3d();

  void createFromShortVolume(const ShortVolume& s, int con);
  void createDualInputFromShortVolume(const ShortVolume& s, const ShortVolume& mask, int con);
  void calculateAttributes(MTAuxData * mta);
  void Filter(double lambda, int filterType);

  void _fillIndexedVolume(maxTreeNode *node, int &counter, int *idxvol);
  int *getIndexedVolume();
  void _fillNodeList(maxTreeNode *node, int &counter, maxTreeNode **nodelist);
  maxTreeNode **getTreeAsArray();
  unsigned short getk(){return _k;}
  void setk(unsigned short val){_k=val;}
   double MTfor3D(char *fname,int *nLeaves); 
//  ShortVolume* isoToShortVolume(short isoval);

  maxTreeNode *getRoot() { return _root; }
  voxelPos *getVoxelList() { return _voxelList; }

  maxTreeNode **getLabels();
  double getLeaves();
  int getNumberOfLeaves() { 
    if (_nrOfLeaves == 0)
      _countLeaves(_root);
    return _nrOfLeaves; 
  }
  
  voxelPos _point2VoxelPos(unsigned long l);
  inline unsigned long _voxelPos2Point(const voxelPos& p){
    return  ((int)p.x) + ( _xSize *  (((int)p.y) + _ySize * ((int) p.z) ) );
  }

  bool getInvertFilterState(){return _invertFilter;}
  void setInvertFilterState(bool state){_invertFilter = state;}

  unsigned long getXSize ();
  unsigned long getYSize ();
  unsigned long getZSize ();
  short getMaxValue ();
  short getMinValue ();
  unsigned long getNumberOfRootVoxels();
  unsigned long getNumberOfNodes();

  static const int FILTER_MIN = 1;
  static const int FILTER_MAX = 2;
  static const int FILTER_DIR = 3;
  static const int FILTER_SUB = 4;
  maxTreeNode *nodeP;
  void _deleteTree();
private:
  /*administration functions */
  void _initVars();
  void _writeNode(ofstream * file, maxTreeNode * node);
  maxTreeNode * _readNode(ifstream * file);
 // void _deleteTree();
  void _doDeleteTree(maxTreeNode *node);

  /*creation */
  int _flood(short h, const ShortVolume& s);
  int _floodDualInput(short h, 
		      const ShortVolume& s, 
		      const ShortVolume& m);
  void _makeChildren();
  
  int _getNeighbors(voxelPos p, voxelPos *neighbors, const ShortVolume& s);

  /* filter functions */

  MTAuxData* _doCalculateAttributes(MTAuxData * mta, maxTreeNode * node);
  void _doSetPeakLevels(maxTreeNode *node);

  bool _doFilter(unsigned short kprime, maxTreeNode *node, double lambda, short parentLevel, bool removed, int filterType);

  void _getLabels(maxTreeNode *node, maxTreeNode **labels);

  /* Handy */
  //unsigned long _voxelPos2Point(const voxelPos& p);

  /* private vars */
  voxelPos * _voxelList;
  unsigned long _xSize, _ySize, _zSize, _totalSize; /* image dimensions */
  short _minLevel;
  short _maxLevel;
  short _levels; /* number of levels between max and min */
  maxTreeNode *_root; /* root node of maxtree */
  bool _invertFilter; /* controls filtering 
                         (attr > lambda) -> (attr <lambda) */ 
  int _nrOfLeaves;
  void _updateTree(maxTreeNode *node);
  void _countLeaves(maxTreeNode *node);
  void _getLeaves(maxTreeNode *node, maxTreeNode **leaves, int &idx, double &quality);
  void getNumberOfNodes(maxTreeNode *node, unsigned long &counter);
  unsigned short _k;

  // enkele wat globalere vars voor maxtreegeneratie
  int _connectivity; /* The amount of pixels in the neigborhood for flooding */
  unsigned long _handledVoxels;
  long _lastPercentage;
  unsigned long * _voxelOffsetAtLevel;
  unsigned char *_status; 
  const static unsigned char ST_NotAnalyzed = 0;
  const static unsigned char ST_InTheQueue = 1;
  const static unsigned char ST_Analyzed = 2;
  unsigned long * _numberOfNodes; /* number of nodes C^h_k at level h */
  maxTreeNode ** _nodeAtLevel; /* If there is a to be proccessed node at that level h */
  
  Queue *_hQueue; 

  vector<maxTreeNode*> _nodeList; /* temporary list to store nodes. to determin children etc */

};

#endif /* MaxTree3D_h */







