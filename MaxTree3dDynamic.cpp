#include <stdlib.h>
#include "MaxTree3d.h"
//#include "MW_Exceptions.h"


/*
 * Constructors
 */
MaxTree3d::MaxTree3d() {
  _initVars();
}
MaxTree3d::MaxTree3d(const MaxTree3d& mt) {
}

MaxTree3d::MaxTree3d(const ShortVolume& s, int con) {
  _initVars();
  createFromShortVolume(s, con);
}

/*
 * Destructors
 */
MaxTree3d::~MaxTree3d() {
  _deleteTree();
  //if (_hQueue!=NULL) free(_hQueue);
}
/*
 * Public methoden
 */


void MaxTree3d::createFromShortVolume(const ShortVolume& s, int con ){
  voxelPos minPosition;

  //eventueel eerst nog kijken of er al een boom is en deze weghalen.


  if (_root != NULL) {
    _deleteTree();
  }

  _xSize=s.getXSize();
  _ySize=s.getYSize();
  _zSize=s.getZSize();
  _totalSize=_xSize * _ySize * _zSize;
  _voxelList=new voxelPos[_totalSize];

  _handledVoxels=1; // Want standaard wordt minimum in de queue gegooid
  _lastPercentage=0;
  if (con <= 6) {
    con = 6;
  } else if (con <= 26) {
    con = 26;
  } else {
    con = 124;
  }
  _connectivity = con;  //6, 26 , 124
  s.getMinMax(_minLevel,_maxLevel);
  _levels=_maxLevel-_minLevel;
  minPosition = s.getMinPosition();

  
  _status = new unsigned char[_totalSize];
  for (unsigned long i=0;i<_totalSize;i++) _status[i] = ST_NotAnalyzed;
  //if (_hQueue!=NULL) free(_hQueue);
  //_hQueue = (Queue<voxelPos> *) new (Queue<voxelPos>)[_levels + 1];
  _hQueue = (Queue *) malloc( (_maxLevel+1)*sizeof(Queue));
 // if (_hQueue==NULL){throw new MW_MemoryAllocException();}
  _numberOfNodes = new unsigned long[_maxLevel + 1];
  
  //_nodeAtLevel = (maxTreeNode **) new (maxTreeNode*)[_maxLevel + 1];

  _nodeAtLevel = (maxTreeNode **) malloc(sizeof(maxTreeNode*)*(_maxLevel + 1));
 // if (_nodeAtLevel==NULL){throw new MW_MemoryAllocException();}

  _voxelOffsetAtLevel = new unsigned long[_maxLevel + 1];
  unsigned long voxelsTillNow = 0;
  for (short i=0; i<=_maxLevel; i++) {
    _numberOfNodes[i] = 0;
    _nodeAtLevel[i] = NULL;
    _voxelOffsetAtLevel[i] = voxelsTillNow;
    voxelsTillNow += s.getNumberOfVoxelsWithValue(i);
    _hQueue[i].resize(1+s.getNumberOfVoxelsWithValue(i));
  }


  //_root = new maxTreeNode;
  _root = (maxTreeNode *) malloc(sizeof(maxTreeNode));
 // if (_root==NULL){throw new MW_MemoryAllocException();}
  
  _root->numberOfVoxels = 0;
  _root->numberOfChildren = 0;
  _root->voxelOffset = 0;
  _root->children = NULL;
  _root->numberOfEdges = 0;
  _root->edgeList = NULL;
  _root->parent=_root;
  
  _nodeAtLevel[_minLevel] = _root;
  _hQueue[_minLevel].push(minPosition);
  _status[_voxelPos2Point(minPosition)] = ST_InTheQueue;

 /* if (mbox==NULL)
    cout << "start flooding" << endl;
  else
    mbox->post("start flooding");

  if (rep!=0){
    rep->label("Flooding Max-tree");
    rep->minimum(0.0);
    rep->maximum(100.0);
    rep->resetProgress();
  }*/

  _lastPercentage = 0;
  _flood(_minLevel, s);


/*  if (mbox==NULL)
    cout << "flooding done" << endl;
  else
    mbox->post("flooding done");*/

  delete[] _status;
  for (short i=0; i<=_maxLevel; i++) {
    _hQueue[i].freeData();
  }
  free(_hQueue);
  _hQueue=NULL;
  delete[] _numberOfNodes;
  free(_nodeAtLevel);

  /*if (rep!=0){
    rep->label("Computing child pointers");
    rep->minimum(0.0);
    rep->maximum(100.0);
    rep->resetProgress();
  }*/
  _makeChildren();
  _doSetPeakLevels(_root);
  
  _nodeList.clear();

  _handledVoxels=0;
  delete[] _voxelOffsetAtLevel;
}




void MaxTree3d::calculateAttributes(MTAuxData * mta ){
  _handledVoxels = 0;
  _lastPercentage = 0;
  /*if (rep!=0){
    rep->label("Computing attribute values");
    rep->minimum(0.0);
    rep->maximum(100.0);
    rep->resetProgress();
  }*/

  MTAuxData *  a = _doCalculateAttributes(mta,_root);
  delete a;
}


void MaxTree3d::Filter(double lambda, int filterType) {
  
  _doFilter(0,_root,lambda,0,false,filterType);
  
}



unsigned long MaxTree3d::getXSize () {
  return _xSize;
}
unsigned long MaxTree3d::getYSize () {
  return _ySize;
}
unsigned long MaxTree3d::getZSize () {
  return _zSize;
}
short MaxTree3d::getMaxValue () {
  return _maxLevel;
}
short MaxTree3d::getMinValue () {
  return _minLevel;
}

unsigned long MaxTree3d::getNumberOfRootVoxels(){
  if (_root==NULL) return 0;
  else return _root->numberOfVoxels;
}


/* 
 * private methoden
 */

/* admin */
void MaxTree3d::_initVars() {
  _nrOfLeaves = 0;
  _root = NULL;
  _hQueue = NULL;
  _invertFilter = 0;
  _k=0;
}

void MaxTree3d::_writeNode(ofstream * file, maxTreeNode * node){
  unsigned long i;
  unsigned char byteValue;
  unsigned long longValue;
  short shortValue;
  //schrijf attribute ??

  //schrijf level
  shortValue = node->level;
  file->put((char)(shortValue >> 8));
  file->put((char)(shortValue & 0xFF));
  //schrijf currentValue
  shortValue = node->currentValue;
  file->put((char)(shortValue >> 8));
  file->put((char)(shortValue & 0xFF));
  //schrijf voxelOffset
  longValue = node->voxelOffset;
  for (i=0;i<4;i++) {
    byteValue = (longValue & (0xFF000000 >> (8*i))) >> ((3-i)*8);
    file->put(byteValue);
  }
  //schrijf numberOfVoxels
  longValue = node->numberOfVoxels;
  for (i=0;i<4;i++) {
    byteValue = (longValue & (0xFF000000 >> (8*i))) >> ((3-i)*8);
    file->put(byteValue);
  }
  //schrijf numberOfChildren
  longValue = node->numberOfChildren;
  for (i=0;i<4;i++) {
    byteValue = (longValue & (0xFF000000 >> (8*i))) >> ((3-i)*8);
    file->put(byteValue);
  }
  //en kindjes wegschrijven
  for (i=0; i < node->numberOfChildren ; i++) {
    _writeNode(file,node->children[i]);
  }
}
maxTreeNode * MaxTree3d::_readNode(ifstream * file){
  unsigned long i;
  char byteValue;

  //maxTreeNode * node = new maxTreeNode;
  maxTreeNode * node = (maxTreeNode *) malloc(sizeof(maxTreeNode));
  node->parent = NULL;
  node->children = NULL;
  //lees attribute ??
  node->attribute = 0.0;
  //lees level
  file->get(byteValue);
  node->level=((unsigned char) byteValue);
  node->level <<= 8;
  file->get(byteValue);
  node->level+=((unsigned char) byteValue);
  //lees currentValue
  file->get(byteValue);
  node->currentValue=((unsigned char) byteValue);
  node->currentValue <<= 8;
  file->get(byteValue);
  node->currentValue+=((unsigned char) byteValue);
  //lees voxelOffset
  file->get(byteValue);
  node->voxelOffset=((unsigned char) byteValue);
  for (i=0;i<3;i++) {
    node->voxelOffset <<= 8;
    file->get(byteValue);
    node->voxelOffset += ((unsigned char) byteValue);
  }
  //lees numberOfVoxels
  file->get(byteValue);
  node->numberOfVoxels=((unsigned char) byteValue);
  for (i=0;i<3;i++) {
    node->numberOfVoxels <<= 8;
    file->get(byteValue);
    node->numberOfVoxels += ((unsigned char) byteValue);
  }  
  //lees numberOfChildren
  file->get(byteValue);
  node->numberOfChildren=((unsigned char) byteValue);
  for (i=0;i<3;i++) {
    node->numberOfChildren <<= 8;
    file->get(byteValue);
    node->numberOfChildren += ((unsigned char) byteValue);
  }
  //kind lijst aanmaken
  //node->children = (maxTreeNode **) new (maxTreeNode*)[node->numberOfChildren];
  node->children = (maxTreeNode **) malloc(sizeof(maxTreeNode*) * (node->numberOfChildren));
  //en kindjes wegschrijven
  for (i=0; i < node->numberOfChildren ; i++) {
    node->children[i]=_readNode(file);
    node->children[i]->parent=node;
  }
  return node;
}

void MaxTree3d::_deleteTree(){
  if (_root != NULL) {
    _doDeleteTree(_root);
    delete[] _voxelList;
  }
}

void MaxTree3d::_doDeleteTree(maxTreeNode *node){
  for(unsigned int i=0;i<node->numberOfChildren;i++){
    _doDeleteTree(node->children[i]);
  }
  if (node->numberOfChildren>0)
    free(node->children);
  if (node->edgeList != NULL)
    delete[] node->edgeList;
  free(node);
}

#if 1
/* Creation */
int MaxTree3d::_flood(short h, const ShortVolume& s ) {
  voxelPos *neighbors;
  int numOfNeighbors, i;
  voxelPos p, q;
  short m, atlevel_idx;
  unsigned long q_index;

  neighbors = new voxelPos[_connectivity+1]; // +1 omdat de voxel zelf ook wordt opgeslagen :)
  //staan er op niveau h nog voxels in de queue
  while(!_hQueue[h].empty()) {
    //cout << "Queue (1) size: " << _hQueue[1].size() << endl;
    //cout << "Queue front at level: " << h << endl;
    //cout << "Queue size: " << _hQueue[h].size() << endl;
    p = _hQueue[h].front();
    //cout << "p: " << p.x << " " << p.y << " " << p.z << endl;
    _hQueue[h].pop();
    // aan de node toevoegen
    _voxelList[_voxelOffsetAtLevel[h]]=p;
    _nodeAtLevel[h]->numberOfVoxels++;
    _voxelOffsetAtLevel[h]++;
    //status huidige voxel aanpassen
    _status[_voxelPos2Point(p)] = ST_Analyzed;
    // SOME OUTPUT SHIT
    _handledVoxels++;
  /*  if (rep!=0){
      if ( _handledVoxels > ((_lastPercentage*_totalSize )/100)) {
          _lastPercentage++;
          rep->postProgress((float)_lastPercentage);
	    }
     }*/
     // einde SOME OUTPUT SHIT


    // buren ophalen en bijlangs gaan
    
    numOfNeighbors = _getNeighbors(p, neighbors, s);
    for (i=0; i<numOfNeighbors; i++) {
      q = neighbors[i];
      q_index = _voxelPos2Point(q);
      if (_status[q_index]==ST_NotAnalyzed) {
        // als ie nog niet is geanalyserd, toevoegen aan de queue op z'n level
	    m = s.getVoxel(q);
	    atlevel_idx = m;
	    //cout << "Flood: push at level: " << atlevel_idx << endl;
        _hQueue[atlevel_idx].push(q);
        //status aanpassen
        _status[q_index]=ST_InTheQueue;
        
	    if (_nodeAtLevel[atlevel_idx] == NULL) {
	      //_nodeAtLevel[atlevel_idx] = new maxTreeNode;
	      _nodeAtLevel[atlevel_idx] = (maxTreeNode *) malloc(sizeof(maxTreeNode));
	      _nodeAtLevel[atlevel_idx]->voxelOffset = _voxelOffsetAtLevel[atlevel_idx];
          _nodeAtLevel[atlevel_idx]->numberOfVoxels = 0; 
          _nodeAtLevel[atlevel_idx]->numberOfChildren = 0; 
	      _nodeAtLevel[atlevel_idx]->numberOfEdges = 0;
	      _nodeAtLevel[atlevel_idx]->edgeList = NULL;

	      _nodeList.push_back(_nodeAtLevel[atlevel_idx]);
        }
        
        if (s.getVoxel(q) > s.getVoxel(p)) {
          //Als er een hogere dan de huidige waarde is, dan krijgen we een nieuwe node
          m = s.getVoxel(q);
          do {
            // Net zolang flooden totdat alle bovenliggende pixels in dit gebied zijn geweest
            m = _flood(m, s);
          } while (m!=h);
        } // einde if level
      } //einde if status
    } //einde loop neighbors
  } //einde lus queue

  //cout << "end while" << endl;

  // Zoeken naar de maximale waarde die nog in de queue zit.
  m = h-1;
  while ((m >= _minLevel) &&  (_nodeAtLevel[m]==NULL)) { 
    m--;
  }
  if (m >= _minLevel) {
    _nodeAtLevel[h]->parent = _nodeAtLevel[m];
  }
  //afronden
  _nodeAtLevel[h]->level = h;
  _nodeAtLevel[h]->currentValue = h;
  _nodeAtLevel[h]=NULL; // deze node is klaar dus odm geen node op dit niveau
  delete[] neighbors;
  _numberOfNodes[h]++;
  return m;
}
#endif


#if 0
int MaxTree3d::_flood(short h, const ShortVolume& s) {
  voxelPos *neighbors;
  int numOfNeighbors, i;
  voxelPos p, q;
  short m;
  unsigned long q_index;

  neighbors = new voxelPos[_connectivity+1]; // +1 omdat de voxel zelf ook wordt opgeslagen :)
  //staan er op niveau h nog voxels in de queue
  while(!_hQueue[h].empty()) {
    p = _hQueue[h].front();
    _hQueue[h].pop();
    // aan de node toevoegen
    _voxelList[_voxelOffsetAtLevel[h]]=p;
    _nodeAtLevel[h]->numberOfVoxels++;
    _voxelOffsetAtLevel[h]++;
    //status huidige voxel aanpassen
    _status[_voxelPos2Point(p)] = ST_Analyzed;
    // buren ophalen en bijlangs gaan
    numOfNeighbors = _getNeighbors(p, neighbors, s);
    for (i=0; i<numOfNeighbors; i++) {
      q = neighbors[i];
      q_index = _voxelPos2Point(q);
      if (_status[_voxelPos2Point(q)]==ST_NotAnalyzed) {
        // SOME OUTPUT SHIT
        _handledVoxels++;
        // einde SOME OUTPUT SHIT
        // als ie nog niet is geanalyserd, toevoegen aan de queue op z'n level
	m = s.getVoxel(q);
        _hQueue[s.getVoxel(q)].push(q);
        //status aanpassen
        _status[_voxelPos2Point(q)]=ST_InTheQueue;
	if (_nodeAtLevel[s.getVoxel(q)] == NULL) {
	  _nodeAtLevel[s.getVoxel(q)] = new maxTreeNode;
	  _nodeAtLevel[s.getVoxel(q)]->voxelOffset = _voxelOffsetAtLevel[s.getVoxel(q)];
          _nodeAtLevel[s.getVoxel(q)]->numberOfVoxels = 0; 
          _nodeAtLevel[s.getVoxel(q)]->numberOfChildren = 0; 
	  _nodeList.push_back(_nodeAtLevel[s.getVoxel(q)]);
	}
        if (s.getVoxel(q) > s.getVoxel(p)) {
          //Als er een hogere dan de huidige waarde is, dan krijgen we een nieuwe node
          m = s.getVoxel(q);
          do {
            // Net zolang flooden totdat alle bovenliggende pixels in dit gebied zijn geweest
            m = _flood(m, s);
          } while (m!=h);
        } // einde if level
      } //einde if status
    } //einde loop neighbors
  } //einde lus queue
  // Zoeken naar de maximale waarde die nog in de queue zit.
  m = h-1;
  while ((m >= _minLevel) &&  (_nodeAtLevel[m]==NULL)) { 
    m--;
  }
  if (m >= _minLevel) {
    _nodeAtLevel[h]->parent = _nodeAtLevel[m];
  }
  //afronden
  _nodeAtLevel[h]->level = h;
  _nodeAtLevel[h]->currentValue = h;
  _nodeAtLevel[h]=NULL; // deze node is klaar dus odm geen node op dit niveau
  delete[] neighbors;
  _numberOfNodes[h]++;
  return m;
}
#endif


void MaxTree3d::_makeChildren(){

  unsigned long totalChildNodes = _nodeList.size();
  unsigned long step = (3*totalChildNodes)/100,j=0,pct=0; 
  /*if (rep!=0)
    rep->label("Computing number of child pointers");
  */
  // aantal child values bepalen
  for (unsigned long i=0;i < totalChildNodes;i++,j++){
    if (_nodeList[i]!=0)
      _nodeList[i]->parent->numberOfChildren ++;
   // if (rep!=0)
      if (j==step){pct++; j=0; }
  }

  // array maken en totaal als index gebruiken (dus 0)
  //_root->children = (maxTreeNode **) new (maxTreeNode*)[_root->numberOfChildren];
  _root->children = (maxTreeNode **) malloc(sizeof(maxTreeNode*)*(_root->numberOfChildren));

  _root->numberOfChildren = 0;
/*  if (rep!=0)
    rep->label("Computing allocating pointers");
*/
  for (unsigned long i=0;i < totalChildNodes;i++,j++){
    //_nodeList[i]->children = (maxTreeNode **) new (maxTreeNode*)[_nodeList[i]->numberOfChildren];
    _nodeList[i]->children = (maxTreeNode **) malloc(sizeof(maxTreeNode*)*(_nodeList[i]->numberOfChildren));
    _nodeList[i]->numberOfChildren = 0;
    
    /*if (rep!=0)
      if (j==step){pct++; j=0; rep->postProgress((float)pct); }*/
  }
  /*if (rep!=0)
    rep->label("Assigning child pointers");*/

  // kindertjes invoegen
  for (unsigned long i=0;i < totalChildNodes;i++,j++){
    _nodeList[i]->parent->children[_nodeList[i]->parent->numberOfChildren] = _nodeList[i];
    _nodeList[i]->parent->numberOfChildren ++;

 //   if (rep!=0)
      if (j==step){pct++; j=0; }
  }
  /*  if (rep!=0)
      rep->postProgress(100.0);*/
}



int MaxTree3d::_getNeighbors(voxelPos p, voxelPos *neighbors, const ShortVolume& s){
  int n=0;
  voxelPos tmp;
  long xx,yy,zz;
  long xlb,ylb,zlb,xub,yub,zub;

  switch (_connectivity){
  case 6:
    tmp.x=p.x+1;
    tmp.y=p.y;
    tmp.z=p.z;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    tmp.x=p.x-1;
    tmp.y=p.y;
    tmp.z=p.z;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    tmp.x=p.x;
    tmp.y=p.y+1;
    tmp.z=p.z;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    tmp.x=p.x;
    tmp.y=p.y-1;
    tmp.z=p.z;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    tmp.x=p.x;
    tmp.y=p.y;
    tmp.z=p.z+1;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    tmp.x=p.x;
    tmp.y=p.y;
    tmp.z=p.z-1;
    if (s.inBounds(tmp)){
      neighbors[n]=tmp;
      n++;
    }
    
    break;

  case 26:
     
    xlb = (p.x > 0) ? (p.x-1) : 0;
    ylb = (p.y > 0) ? (p.y-1) : 0;
    zlb = (p.z > 0) ? (p.z-1) : 0;
    xub = (p.x < (_xSize-1))  ? p.x+1 : _xSize-1;
    yub = (p.y < (_ySize-1)) ? p.y+1 : _ySize-1;
    zub = (p.z < (_zSize-1))  ? p.z+1 : _zSize-1;

    for (zz=zlb; zz<=zub; zz++) 
      for (yy=ylb; yy<=yub; yy++) 
	    for (xx=xlb; xx<=xub; xx++) { 
	      tmp.x = xx;
	      tmp.y = yy;
	      tmp.z = zz;
	      neighbors[n] = tmp;
	      n++;
  	    }
    

    break;
  case 124:
    for (int i=-2;i<3;i++){
      for (int j=-2;j<3;j++){
        for (int k=-2;k<3;k++){
          tmp.x=p.x+i;
          tmp.y=p.y+j;
          tmp.z=p.z+k;
          if (s.inBounds(tmp)){
            neighbors[n]=tmp;
            n++;
          }
        }
      }
    }
    break;
  }

  return n;
}


voxelPos MaxTree3d::_point2VoxelPos(unsigned long l){
  voxelPos v;
  unsigned long xx = l;
  unsigned long yy;
#if 0
  v.x = l;
  v.y = v.x / _xSize;
  v.z = v.y / _ySize;
  
  v.x -= v.y * _xSize;
  v.y -= v.z * _ySize;
#endif
  yy = xx / _xSize;
  v.z = yy / _ySize;
  
  v.x = xx - yy * _xSize;
  v.y = yy - v.z * _ySize;

  return v;
}

/* Filter functions */

MTAuxData* MaxTree3d::_doCalculateAttributes(MTAuxData * mta, maxTreeNode * node){
  MTAuxData * attr;
  MTAuxData * childattr;
  voxelPos vp;

  attr = mta->getNewInstance();
  //voxels toevoegen
  for (unsigned int i=0;i<node->numberOfVoxels;i++) {
    vp = _voxelList[node->voxelOffset + i];
    attr->addData(vp.x,vp.y,vp.z);
    _handledVoxels++;
   /* if (rep!=0){
      if (((_handledVoxels*100)/_totalSize) > (_lastPercentage )) {
          _lastPercentage=(_handledVoxels*100)/_totalSize;
          rep->postProgress((float)_lastPercentage);
	    }
     }*/
    
  }
  //kinderen bijlangsgaan
  for(unsigned int i=0;i<node->numberOfChildren;i++){
    childattr = _doCalculateAttributes(mta, node->children[i]);
    attr->mergeAuxData(childattr);
    delete childattr;
  }
  node->attribute = attr->getAttribute();
  return attr;  
}

void MaxTree3d::_doSetPeakLevels(maxTreeNode *node){
  node->peakLevel=node->level;
  if (node->numberOfChildren > 0){
     for(unsigned int i=0;i<node->numberOfChildren;i++){
       _doSetPeakLevels(node->children[i]);
       if (node->children[i]->peakLevel > node->peakLevel)
         node->peakLevel=node->children[i]->peakLevel;
     }      
  }     
}

void kCorrection(maxTreeNode *node, unsigned short k, unsigned short parentlevel){
  if (node->currentValue != node->level){
     int difflevel= node->level - parentlevel;
     node->currentValue = (difflevel>k) ? parentlevel + k : parentlevel + difflevel;
     for (int i = 0; i<node->numberOfChildren;i++){
       kCorrection(node->children[i],k,parentlevel);
     }
  }  
}

bool MaxTree3d::_doFilter( unsigned short kprime, 
                            maxTreeNode *node, 
                            double lambda, 
                            short parentLevel, 
                            bool removed, 
                            int filterType){
  //vergelijk lambda met attribute value
  unsigned long i;
  unsigned short s = _minLevel;
  bool result = true;
  bool postRecursion = true;
  maxTreeNode *parent = node->parent;
  unsigned short difflevel  = node->level - parent->level;
  if (difflevel==0) difflevel=_minLevel;
  switch(filterType) {
    case FILTER_MIN:
      /* de boolean removed geeft aan of 1 van de voorouders al is
	 verwijderd. Waneer dit het geval is wordt de rest van de tak
	 op het niveau van de laatst getekende getekend */
      if (removed) {
         if (difflevel>=kprime) {
            s = parentLevel + kprime;
            kprime=0;
         } else {
           s = parentLevel + difflevel;
           kprime -= difflevel;
         }
      } else if ((node->peakLevel - parentLevel > _k) &&
          ((node->attribute > lambda) != _invertFilter)) {
	    s=node->level;
	    kprime = _k;
      } else {
    	removed = true;
         if (difflevel>=kprime) {
            s = parentLevel + kprime;
            kprime=0;
         } else {
           s = parentLevel + difflevel;
           kprime -= difflevel;
         }
      }
      parentLevel = s;
      break;
    case FILTER_MAX:
      /* parentlevel geeft ook hier het laatst getekende level
	  weer. Grote verschil hier is dat hier eerst wordt gekeken of
	  er nog getekende kinderen zijn.*/
      postRecursion = false;
      node->currentValue = 0;
      s=0;
      result = false;
      for (i=0;i<node->numberOfChildren;i++){
        result = _doFilter(kprime,node->children[i],lambda,parentLevel,removed,filterType) || result;
      }
      if (result){
        s = node->level;
        node->currentValue = s;
        for (i=0;i<node->numberOfChildren;i++){
          kCorrection(node->children[i],_k,node->level);  
        }
      } else {
        if ((node->peakLevel - parent->level > _k) &&
          ((node->attribute > lambda) != _invertFilter)){ // sowieso tekenen
	      result = true;
	      s = node->level;
          node->currentValue = s;
	      for (i=0;i<node->numberOfChildren;i++){
             kCorrection(node->children[i],_k,node->level);   
          }
        } else {
          s=0;
        }
      }
      break;
    case FILTER_DIR:
      /* parentLevel wordt gebruikt om het vorig -getekende- parent
	 niveau te onthouden. Waneer een node niet wordt getekend,
	 krijgt deze de kleur van de laatst getekende parent */
      if ((node->peakLevel - parentLevel > _k) &&
          ((node->attribute > lambda) != _invertFilter)) {
	     s = node->level;
  	     kprime = _k;
      } else {
         if (difflevel>=kprime) {
            s = parentLevel + kprime;
            kprime=0;
         } else {
           s = parentLevel + difflevel;
           kprime -= difflevel;
         }
      }
      parentLevel=s;
      break;
    case FILTER_SUB:
      /* d wordt gebruikt om het verschil in oorspronkelijke level en
	 huidige level bij te houden. Hierin staat dus per tak als het
	 ware de hoogte waarmee een tak zakt.*/
      if ((node->peakLevel - parentLevel > _k) &&
          ((node->attribute > lambda) != _invertFilter)) {
         s = parentLevel + difflevel;
         kprime = _k;
      } else {
         if (difflevel>=kprime) {
            s = parentLevel + kprime;
            kprime=0;
         } else {
           s = parentLevel + difflevel;
           kprime -= difflevel;
         }
      }
      parentLevel=s;
      break;
  default:
    break;
  }
  node->currentValue = s;
  
  if (postRecursion) 
    for (i=0;i<node->numberOfChildren;i++){
      _doFilter(kprime,node->children[i],lambda,parentLevel,removed,filterType);
  }
  return result;
}


void MaxTree3d::_getLabels(maxTreeNode *node, maxTreeNode **labels)
{
  voxelPos vp;

  for (unsigned int i=0;i<node->numberOfVoxels;i++) {
    vp = _voxelList[node->voxelOffset + i];
    labels[_voxelPos2Point(vp)] = node;
    _handledVoxels ++;
    /*if ( (rep!=0) && (((100*_handledVoxels)/_totalSize) > _lastPercentage)){
      _lastPercentage = (100*_handledVoxels)/_totalSize;
      rep->postProgress((float) _lastPercentage);
    } */
  }

  for(unsigned int j=0;j<node->numberOfChildren;j++){
    _getLabels(node->children[j], labels);
  }
}


maxTreeNode **MaxTree3d::getLabels()
{

  //maxTreeNode *dummy = new maxTreeNode();
  //maxTreeNode *dummy = (maxTreeNode *) malloc(sizeof(maxTreeNode));
  //dummy->level = 32000;

  //maxTreeNode **labels = (maxTreeNode **) new (maxTreeNode*)[_xSize * _ySize * _zSize];
  maxTreeNode **labels = (maxTreeNode **) malloc(sizeof(maxTreeNode*)*(_xSize * _ySize * _zSize));
  //for (int i=0; i<_xSize * _ySize * _zSize; i++)
  //  labels[i] = dummy;
  _handledVoxels=0;
  _lastPercentage=0;
 /* if (rep!=0){
    rep->label("Getting labels");
    rep->minimum(0.0);
    rep->maximum(100.0);
  }*/
  _getLabels(_root, labels);
  return labels;
}


void MaxTree3d::_fillIndexedVolume(maxTreeNode *node, int &counter, 
				   int *idxvol)
{

  counter++;

  voxelPos vp;

  for (unsigned int i=0;i<node->numberOfVoxels;i++) {
    vp = _voxelList[node->voxelOffset + i];
    idxvol[_voxelPos2Point(vp)] = counter;
  }


  for (unsigned int i=0; i<node->numberOfChildren; i++)
    _fillIndexedVolume(node->children[i], counter, idxvol);


}

int *MaxTree3d::getIndexedVolume()
{

  int *idxvol = new int[_xSize * _ySize * _zSize];

  int counter = 0;
  _fillIndexedVolume(_root, counter, idxvol);

  return idxvol;

}

void MaxTree3d::_fillNodeList(maxTreeNode *node, int &counter, 
			      maxTreeNode **nodelist)
{

  counter++;
  nodelist[counter] = node;
  
  for (unsigned int i=0; i<node->numberOfChildren; i++)
    _fillNodeList(node->children[i], counter, nodelist);
}

maxTreeNode **MaxTree3d::getTreeAsArray()
{

  int nrNodes = getNumberOfNodes();
  maxTreeNode **nodelist = new maxTreeNode*[nrNodes + 1];
  nodelist[0] = 0;
  int counter = 0;
  _fillNodeList(_root, counter, nodelist);

#if 0
  for (int i=1; i<=nrNodes; i++)
    std::cout << "Node " << i << " -> " << nodelist[i]->currentValue << std::endl; 
#endif

  return nodelist;
}


void MaxTree3d::_countLeaves(maxTreeNode *node)
{
  if (node->numberOfChildren == 0)
    _nrOfLeaves++;

  for (unsigned int i=0; i<node->numberOfChildren; i++)
    _countLeaves(node->children[i]);
}

void MaxTree3d::_updateTree(maxTreeNode *node)
{
maxTreeNode *nodeP;
int cnt;

  if (node->numberOfChildren == 0) {
    if(node->Dynamic<0.05)
      {
      nodeP= node->parent;
	for (unsigned int k=0; k<nodeP->numberOfChildren; k++)
            if(nodeP->children[k]->Dynamic<0.05)
               cnt++;
        if(cnt==nodeP->numberOfChildren)
          nodeP->numberOfChildren=0; 
      }
    
  }

  for (unsigned int i=0; i<node->numberOfChildren; i++)
    _updateTree(node->children[i]);

}
void MaxTree3d::_getLeaves(maxTreeNode *node, maxTreeNode **leaves, int &idx, double &quality)
{
maxTreeNode *nodeP,*nodePold;
unsigned long idx1;
bool fork=0;
//double quality;

  if (node->numberOfChildren == 0) {
    leaves[idx] = node;
      
      nodeP= node->parent;
      
    while(!fork)
    {      
     //printf("Here\n");
      if (nodeP->numberOfChildren>1)
		 { 
		    node->Dynamic=((double)(node->level-nodeP->level)/(double)(node->level));
                    if(node->Dynamic>0.05)
                    quality=quality+node->Dynamic;
                    fork=1;
		 }
	  
        else if(nodeP==_root)
	        {  fork=1; quality=0;}
		 nodeP= nodeP->parent;
         }
     
    
    idx++;
  }

  for (unsigned int i=0; i<node->numberOfChildren; i++)
    _getLeaves(node->children[i], leaves, idx,quality);

}

double MaxTree3d::getLeaves()
{
double quality=0.0;
  if (_nrOfLeaves == 0)
    _countLeaves(_root);

  int idx = 0;
  //maxTreeNode **leaves = (maxTreeNode **) new (maxTreeNode *)[_nrOfLeaves];
  maxTreeNode **leaves = (maxTreeNode **) malloc(sizeof(maxTreeNode *)*(_nrOfLeaves));
  _getLeaves(_root, leaves, idx,quality);
  _updateTree(_root);
   _getLeaves(_root, leaves, idx,quality);
  return quality;

}


unsigned long MaxTree3d::getNumberOfNodes()
{

  unsigned long counter = 0;
  getNumberOfNodes(_root, counter);

  return counter;
}


void MaxTree3d::getNumberOfNodes(maxTreeNode *node, unsigned long &counter)
{
  
  counter++;
  for (unsigned int i=0; i<node->numberOfChildren; i++)
    getNumberOfNodes(node->children[i], counter);

}
