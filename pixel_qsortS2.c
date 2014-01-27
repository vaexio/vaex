/*----------------------------------------------------------------------------
   Function :   pixel_qsort()
   In       :   pixel array, size of the array
   Out      :   void
   Job      :   sort out the array of pixels
   Notice   :   optimized implementation, unreadable.
 ---------------------------------------------------------------------------*/
#include <string.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

typedef double pixelvalue ;
#define PIX_SWAP(a,b,ia,ib) { pixelvalue temp=(a);(a)=(b);(b)=temp; char tempi=(ia);(ia)=(ib);(ib)=tempi;}
#define PIX_STACK_SIZE 50

void pixel_qsort(pixelvalue *pix_arr,char *index, int npix)
{
    int         i,
                ir,
                j,
                k,
                l;
    int *       i_stack ;
    int         j_stack,ind ;
    pixelvalue  a ;

    ir = npix ;
    l = 1 ;
    j_stack = 0 ;
    i_stack = malloc(PIX_STACK_SIZE * sizeof(pixelvalue)) ;
    for (;;) {
        if (ir-l < 7) {
            for (j=l+1 ; j<=ir ; j++) {
                a = pix_arr[j-1];
                ind=index[j-1];
                for (i=j-1 ; i>=1 ; i--) {
                    if (pix_arr[i-1] <= a) break;
                    pix_arr[i] = pix_arr[i-1];
                    index[i]=index[i-1];
                }
                pix_arr[i] = a;
                index[i]=ind;
            }
            if (j_stack == 0) break;
            ir = i_stack[j_stack-- -1];
            l  = i_stack[j_stack-- -1];
        } else {
            k = (l+ir) >> 1;
            PIX_SWAP(pix_arr[k-1], pix_arr[l],index[k-1], index[l])
            if (pix_arr[l] > pix_arr[ir-1]) {
                PIX_SWAP(pix_arr[l], pix_arr[ir-1],index[l], index[ir-1])
            }
            if (pix_arr[l-1] > pix_arr[ir-1]) {
                PIX_SWAP(pix_arr[l-1], pix_arr[ir-1],index[l-1], index[ir-1])
            }
            if (pix_arr[l] > pix_arr[l-1]) {
                PIX_SWAP(pix_arr[l], pix_arr[l-1],index[l], index[l-1])
            }
            i = l+1;
            j = ir;
            a = pix_arr[l-1];
            ind=index[l-1];
            for (;;) {
                do i++; while (pix_arr[i-1] < a);
                do j--; while (pix_arr[j-1] > a);
                if (j < i) break;
                PIX_SWAP(pix_arr[i-1], pix_arr[j-1],index[i-1], index[j-1]);
            }
            pix_arr[l-1] = pix_arr[j-1];
            pix_arr[j-1] = a;
             index[l-1] = index[j-1];
            index[j-1] = ind;
            j_stack += 2;
            if (j_stack > PIX_STACK_SIZE) {
                printf("stack too small in pixel_qsort: aborting");
                exit(-2001) ;
            }
            if (ir-i+1 >= j-l) {
                i_stack[j_stack-1] = ir;
                i_stack[j_stack-2] = i;
                ir = j-1;
            } else {
                i_stack[j_stack-1] = j-1;
                i_stack[j_stack-2] = l;
                l = i;
            }
        }
    }
    free(i_stack) ;
}
#undef PIX_STACK_SIZE
#undef PIX_SWAP
