/*
Please note that this and other following examples were taking from book:
 "programming massively parallel processors: a hands-on approach 4th edition"
*/
#include <stdio.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    for(int i = 0; i<n; i++){
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main(){
    // memory allocation for arrays A, B, and C
    // I/O to read A and B, N elements each...
    float A[] = {1,2,3,4,5};
    float B[] = {6,7,8,9,10};
    float C[5];
    int N = sizeof(A) / sizeof(A[0]);
    
    // call the func
    vecAdd(A,B,C,N);

    // print the array C
    for(int i = 0; i<N; i++){
        printf("%f ", C[i]);
    }
    // output : 7.000000 9.000000 11.000000 13.000000 15.000000 
    return 0;

}