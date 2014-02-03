/*----------------------------------------------------------------------------
   Function :   pixel_qsort()
   In       :   pixel array, size of the array
   Out      :   void
   Job      :   sort out the array of pixels
   Notice   :   optimized implementation, unreadable.
 ---------------------------------------------------------------------------*/

typedef double pixelvalue ;
#define PIX_SWAP(a,b) { pixelvalue temp=(a);(a)=(b);(b)=temp; }
#define PIX_STACK_SIZE 50

void pixel_qsort(pixelvalue *pix_arr, int npix)
{
    int         i,
                ir,
                j,
                k,
                l;
    int *       i_stack ;
    int         j_stack ;
    pixelvalue  a ;

    ir = npix ;
    l = 1 ;
    j_stack = 0 ;
    i_stack = malloc(PIX_STACK_SIZE * sizeof(pixelvalue)) ;
    for (;;) {
        if (ir-l < 7) {
            for (j=l+1 ; j<=ir ; j++) {
                a = pix_arr[j-1];
                for (i=j-1 ; i>=1 ; i--) {
                    if (pix_arr[i-1] <= a) break;
                    pix_arr[i] = pix_arr[i-1];
                }
                pix_arr[i] = a;
            }
            if (j_stack == 0) break;
            ir = i_stack[j_stack-- -1];
            l  = i_stack[j_stack-- -1];
        } else {
            k = (l+ir) >> 1;
            PIX_SWAP(pix_arr[k-1], pix_arr[l])
            if (pix_arr[l] > pix_arr[ir-1]) {
                PIX_SWAP(pix_arr[l], pix_arr[ir-1])
            }
            if (pix_arr[l-1] > pix_arr[ir-1]) {
                PIX_SWAP(pix_arr[l-1], pix_arr[ir-1])
            }
            if (pix_arr[l] > pix_arr[l-1]) {
                PIX_SWAP(pix_arr[l], pix_arr[l-1])
            }
            i = l+1;
            j = ir;
            a = pix_arr[l-1];
            for (;;) {
                do i++; while (pix_arr[i-1] < a);
                do j--; while (pix_arr[j-1] > a);
                if (j < i) break;
                PIX_SWAP(pix_arr[i-1], pix_arr[j-1]);
            }
            pix_arr[l-1] = pix_arr[j-1];
            pix_arr[j-1] = a;
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
