  #include "CImg.h"
  #include<string.h>

  using namespace cimg_library;

  int main() {
    int i=0;
    char fn[30],fnc[30],num[2];
    strcpy(fn,"density");
 
    CImg<unsigned char> image(256,256,1,3,0);
    CImgDisplay main_disp(image,"Click left mouse button to interrupt");
   
    while (main_disp.mouse_y()!=0) {
      
    for(i=0;i<5;i++)
    {
    strcpy(fnc,fn); 
    sprintf(num,"%d",i);
    strcat(fnc,num);
    
    strcat(fnc,".pgm");
  
     image.assign(fnc);
    //main_disp.assign(image);
     image.display(main_disp);
    main_disp.wait(1000);
    //CImgDisplay main_disp(image,"Click a point");
    
     }
    }
    return 0;
  }
