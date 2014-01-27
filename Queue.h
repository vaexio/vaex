#ifndef _Queue
#define _Queue

#include <stdlib.h>

#include "SimpleShortVolume.h"

class Queue {

public:
  Queue() {
    _data = NULL;
    _front = 0;
    _back = 0;
  }

  ~Queue() {
    free(_data);
  }

  void freeData() {
    free(_data);
  }

  int size() {
    return (_back - _front);
  }

  bool empty() {
    return (_front == _back);
  }

  voxelPos front() {
    return _data[_front];
  }

  void pop() {
    _front++;
  }

  void push(const voxelPos elem) {
    _data[_back] = elem;
    _back++;
  }

  void resize(int s) {
    _front = 0;
    _back = 0;
    _size = s;
    //_data = new voxelPos[s];
    _data = (voxelPos *) malloc(s * sizeof(voxelPos));
  }



private:
  int _front, _back;
  voxelPos *_data;
  int _size;

};



#endif
