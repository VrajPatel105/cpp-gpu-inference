#include <iostream>
using namespace std;


// matrix multiplication function here 
void matmul(float* A, float* B, float* bias, float* out, int M, int K, int N){

    for(int m = 0; m<M; m++){
        for(int n = 0; n<N; n++){
            float val = (bias != nullptr) ? bias[n] : 0.0f;
            cout << "second loop" << " n = " << n << " val : " << val << endl;
            for(int k = 0; k<K; k++){
                cout << "third loop " << " k = " << k << endl;
                cout << "val : " << val << " m:" << m << " n:" << n << " k:" << k << " this is before updating the inner loop" << endl;
                val += A[m*K + k] * B[k*N + n];
                cout << "val = " << "A[m*K+k]" << " m:" << m << " N:" << N << " n:" << n << " m*K+k =" << m*K+k << " A[m*K+k]=" << A[m*K+k] << endl;
            }
            out[m*N + n] = val;
            cout << "final value decided " << val << endl;
            cout << "out[m*N + n] = " << "m * N + n = " << m << "*" << N << "+" << n << "=" << out[m*N+n] << endl;
        }
    }

}


int main() {

    float A[] = {1,2,3, 4,5,6, 7,8,9};  // flat 1D array, shape (3,3)
    float B[] = {1,2,3, 4,5,6, 7,8,9};
    float bias[] = {1,2,3};              // bias is 1D, size N
    float out[9] = {0};                  // flat output, shape (3,3), initialized to 0

    matmul(A, B, bias, out, 3, 3, 3);

    for(int i = 0; i<9; i++){
        cout << out[i] << endl;
    }

    return 0;
}

------------------------------------------------------------------------
// why this trick works ?

// summary from ai: 
// WHY FLAT ARRAY + INDEX MATH WORKS:
//
// Memory is just one long line of boxes: [box0, box1, box2, box3, ...]
// There is no "2D" or "3D" in hardware. A matrix is a human concept.
//
// A 3x3 matrix is really just 9 consecutive boxes in memory:
// [1, 2, 3, 4, 5, 6, 7, 8, 9]
//  row0------  row1------  row2------
//
// To find element at row m, column k:
//   - skip m complete rows (each row has K elements) → m * K
//   - move k more steps into that row               → + k
//   - total offset: m * K + k
//
// Same rule for any shape:
//   2D (M, K):       element (m, k)       → m*K + k
//   3D (B, T, C):    element (b, t, c)    → b*T*C + t*C + c
//   4D (B,NH,T,T):   element (b,h,t1,t2) → b*NH*T*T + h*T*T + t1*T + t2
//
// Pattern: each index multiplies by the product of all dimensions to its right.
//
// WHY THIS MATTERS FOR GPU:
// A GPU also has flat memory. Each CUDA thread gets assigned one output
// position (m, n) and uses m*N + n to find where to write its result.
// Same formula, millions of threads running in parallel.
// The math never changes. Only the execution model does.

--------------------------------------------------------------------------------

// // final output 
// second loop n = 0 val : 1
// third loop  k = 0
// val : 1 m:0 n:0 k:0 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:0 m*K+k =0 A[m*K+k]=1
// third loop  k = 1
// val : 2 m:0 n:0 k:1 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:0 m*K+k =1 A[m*K+k]=2
// third loop  k = 2
// val : 10 m:0 n:0 k:2 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:0 m*K+k =2 A[m*K+k]=3
// final value decided 31
// out[m*N + n] = m * N + n = 0*3+0=31
// second loop n = 1 val : 2
// third loop  k = 0
// val : 2 m:0 n:1 k:0 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:1 m*K+k =0 A[m*K+k]=1
// third loop  k = 1
// val : 4 m:0 n:1 k:1 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:1 m*K+k =1 A[m*K+k]=2
// third loop  k = 2
// val : 14 m:0 n:1 k:2 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:1 m*K+k =2 A[m*K+k]=3
// final value decided 38
// out[m*N + n] = m * N + n = 0*3+1=38
// second loop n = 2 val : 3
// third loop  k = 0
// val : 3 m:0 n:2 k:0 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:2 m*K+k =0 A[m*K+k]=1
// third loop  k = 1
// val : 6 m:0 n:2 k:1 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:2 m*K+k =1 A[m*K+k]=2
// third loop  k = 2
// val : 18 m:0 n:2 k:2 this is before updating the inner loop
// val = A[m*K+k] m:0 N:3 n:2 m*K+k =2 A[m*K+k]=3
// final value decided 45
// out[m*N + n] = m * N + n = 0*3+2=45
// second loop n = 0 val : 1
// third loop  k = 0
// val : 1 m:1 n:0 k:0 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:0 m*K+k =3 A[m*K+k]=4
// third loop  k = 1
// val : 5 m:1 n:0 k:1 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:0 m*K+k =4 A[m*K+k]=5
// third loop  k = 2
// val : 25 m:1 n:0 k:2 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:0 m*K+k =5 A[m*K+k]=6
// final value decided 67
// out[m*N + n] = m * N + n = 1*3+0=67
// second loop n = 1 val : 2
// third loop  k = 0
// val : 2 m:1 n:1 k:0 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:1 m*K+k =3 A[m*K+k]=4
// third loop  k = 1
// val : 10 m:1 n:1 k:1 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:1 m*K+k =4 A[m*K+k]=5
// third loop  k = 2
// val : 35 m:1 n:1 k:2 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:1 m*K+k =5 A[m*K+k]=6
// final value decided 83
// out[m*N + n] = m * N + n = 1*3+1=83
// second loop n = 2 val : 3
// third loop  k = 0
// val : 3 m:1 n:2 k:0 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:2 m*K+k =3 A[m*K+k]=4
// third loop  k = 1
// val : 15 m:1 n:2 k:1 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:2 m*K+k =4 A[m*K+k]=5
// third loop  k = 2
// val : 45 m:1 n:2 k:2 this is before updating the inner loop
// val = A[m*K+k] m:1 N:3 n:2 m*K+k =5 A[m*K+k]=6
// final value decided 99
// out[m*N + n] = m * N + n = 1*3+2=99
// second loop n = 0 val : 1
// third loop  k = 0
// val : 1 m:2 n:0 k:0 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:0 m*K+k =6 A[m*K+k]=7
// third loop  k = 1
// val : 8 m:2 n:0 k:1 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:0 m*K+k =7 A[m*K+k]=8
// third loop  k = 2
// val : 40 m:2 n:0 k:2 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:0 m*K+k =8 A[m*K+k]=9
// final value decided 103
// out[m*N + n] = m * N + n = 2*3+0=103
// second loop n = 1 val : 2
// third loop  k = 0
// val : 2 m:2 n:1 k:0 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:1 m*K+k =6 A[m*K+k]=7
// third loop  k = 1
// val : 16 m:2 n:1 k:1 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:1 m*K+k =7 A[m*K+k]=8
// third loop  k = 2
// val : 56 m:2 n:1 k:2 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:1 m*K+k =8 A[m*K+k]=9
// final value decided 128
// out[m*N + n] = m * N + n = 2*3+1=128
// second loop n = 2 val : 3
// third loop  k = 0
// val : 3 m:2 n:2 k:0 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:2 m*K+k =6 A[m*K+k]=7
// third loop  k = 1
// val : 24 m:2 n:2 k:1 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:2 m*K+k =7 A[m*K+k]=8
// third loop  k = 2
// val : 72 m:2 n:2 k:2 this is before updating the inner loop
// val = A[m*K+k] m:2 N:3 n:2 m*K+k =8 A[m*K+k]=9
// final value decided 153
// out[m*N + n] = m * N + n = 2*3+2=153
// 31
// 38
// 45
// 67
// 83
// 99
// 103
// 128
// 153