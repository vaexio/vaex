/*twiddle.c - generate all combinations of M elements drawn without replacement
  from a set of N elements.  This routine may be used in two ways:
  (0) To generate all combinations of M out of N objects, let a[0..N-1]
      contain the objects, and let c[0..M-1] initially be the combination
      a[N-M..N-1].  While twiddle(&x, &y, &z, p) is false, set c[z] = a[x] to
      produce a new combination.
  (1) To generate all sequences of 0's and 1's containing M 1's, let
      b[0..N-M-1] = 0 and b[N-M..N-1] = 1.  While twiddle(&x, &y, &z, p) is
      false, set b[x] = 1 and b[y] = 0 to produce a new sequence.
  In either of these cases, the array p[0..N+1] should be initialised as
  follows:
    p[0] = N+1
    p[1..N-M] = 0
    p[N-M+1..N] = 1..M
    p[N+1] = -2
    if M=0 then p[1] = 1
  In this implementation, this initialisation is accomplished by calling
  inittwiddle(M, N, p), where p points to an array of N+2 ints.

  Coded by Matthew Belmonte <mkb4@Cornell.edu>, 23 March 1996.  This
  implementation Copyright (c) 1996 by Matthew Belmonte.  Permission for use and
  distribution is hereby granted, subject to the restrictions that this
  copyright notice and reference list be included in its entirety, and that any
  and all changes made to the program be clearly noted in the program text.

  This software is provided 'as is', with no warranty, express or implied,
  including but not limited to warranties of merchantability or fitness for a
  particular purpose.  The user of this software assumes liability for any and
  all damages, whether direct or consequential, arising from its use.  The
  author of this implementation will not be liable for any such damages.

  Reference:

  Phillip J Chase, `Algorithm 382: Combinations of M out of N Objects [G6]',
  Communications of the Association for Computing Machinery 13:6:368 (1970).

  The returned indices x, y, and z in this implementation are decremented by 1,
  in order to conform to the C language array reference convention.  Also, the
  parameter 'done' has been replaced with a Boolean return value.
*/
#include<string.h>

int twiddle(x, y, z, p)
int *x, *y, *z, *p;
  {
  register int i, j, k;
  j = 1;
  while(p[j] <= 0)
    j++;
  if(p[j-1] == 0)
    {
    for(i = j-1; i != 1; i--)
      p[i] = -1;
    p[j] = 0;
    *x = *z = 0;
    p[1] = 1;
    *y = j-1;
    }
  else
    {
    if(j > 1)
      p[j-1] = 0;
    do
      j++;
    while(p[j] > 0);
    k = j-1;
    i = j;
    while(p[i] == 0)
      p[i++] = -1;
    if(p[i] == -1)
      {
      p[i] = p[k];
      *z = p[k]-1;
      *x = i-1;
      *y = k-1;
      p[k] = -1;
      }
    else
      {
      if(i == p[0])
	return(1);
      else
	{
	p[j] = p[i];
	*z = p[i]-1;
	p[i] = 0;
	*x = j-1;
	*y = i-1;
	}
      }
    }
  return(0);
  }

void inittwiddle(m, n, p)
int m, n, *p;
  {
  int i;
  p[0] = n+1;
  for(i = 1; i != n-m+1; i++)
    p[i] = 0;
  while(i != n+1)
    {
    p[i] = i+m-n;
    i++;
    }
  p[n+1] = -2;
  if(m == 0)
    p[1] = 1;
  }

/************************/
//  Here is a sample use of twiddle() and inittwiddle():
//#define N 5
//#define M 3
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*(0) To generate all combinations of M out of N objects, let a[0..N-1]
      contain the objects, and let c[0..M-1] initially be the combination
      a[N-M..N-1].  While twiddle(&x, &y, &z, p) is false, set c[z] = a[x] to
      produce a new combination.
 (1) To generate all sequences of 0's and 1's containing M 1's, let
      b[0..N-M-1] = 0 and b[N-M..N-1] = 1.  While twiddle(&x, &y, &z, p) is
      false, set b[x] = 1 and b[y] = 0 to produce a new sequence.*/
//int main()
int combi(int N,int M,int c[][50000])
  {
  int i,j=0,k=1, x, y, z, p[N+2], b[N],tmp[M];
  char a[2],s[2];

  for (i = 0; i < N; ++i) { b[i] = i; }

  inittwiddle(M, N, p);
  int cnt=0;
  for(i = N-M; i<=N-1; i++)
    {
    tmp[j]=b[i];
    if (i==(N-M))
    {
     sprintf(a,"%d",tmp[j]);
     c[cnt][0]=tmp[j];
   // printf("%d\n",c[cnt][0]);
     cnt++;
	}
    else
    {
     sprintf(s,"%d",tmp[j]);
     c[cnt][0]=tmp[j];
    //printf("%d\n",c[cnt][0]);
     cnt++;
     strcat(a,s);
    } 
    j++;
    }
  //c[0][0]=atoi(a);
 // printf("%s\n", a);
 // cnt=1;
  while(!twiddle(&x, &y, &z, p))
    {
    //b[x] = 1;
   // b[y] = 0;
    tmp[z]=b[x];
    for(i=0;i<M;i++)
    {
     if(i==0)
{
       sprintf(a,"%d",tmp[i]);
       c[i][k]=tmp[i];
       //printf("%d\t",c[i][k]);
	cnt++;
}
     else
      {
     sprintf(s,"%d",tmp[i]);
     c[i][k]=tmp[i];
     // printf("%d\n",c[i][k]);
	
     strcat(a,s);
     } 
    // printf("%d",  tmp[i]);
    
    }
   // c[k]=atoi(a);
    k++;
   //printf("\n");
    }
    // printf("NumSpaces:%d\n",k);
return k;//c[M*N][M];
  }
/************************/
