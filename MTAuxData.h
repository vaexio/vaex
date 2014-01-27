#ifndef MTAuxData_h
#define MTAuxData_h


class MTAuxData{
public:

  MTAuxData(){}    
  virtual ~MTAuxData(){} 
  MTAuxData(const MTAuxData& mta){}
  MTAuxData(unsigned long x, unsigned long y, unsigned long z){}
  
  virtual MTAuxData *getNewInstance() const =0;
  virtual void addData(unsigned long x, unsigned long y, unsigned long z) = 0;
  virtual void mergeAuxData(MTAuxData * mta) = 0;
  virtual double getAttribute() = 0;
  virtual void bindVolume(const ShortVolume *vol) = 0;
    
};

#endif /*MTAuxData_h*/
