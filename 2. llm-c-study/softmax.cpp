#include <iostream>
using namespace std;
#include <iterator>
#include <algorithm>
#include <math.h>

void softmax(float* out, float* x, float* weight, float* bias, float eps, int B, int T, int C){
    // argument's info:
    // out -> where to write the result, shape (B,T,C)
    // x -> input, shape (B,T,C). One C-dim vector per token
    // weight -> scale, shape(C,). One value per channel.
    // bias -> shift, shape(C,). One value per channel.
    // eps -> small number like 1e-5, prevents divide by zero in rstd
    // B -> Batch Size
    // T -> sequence length
    // C -> d_model

    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            // lets also get the pointer the the token's C-dim vector
            float* x_bt = x + b*T*C + t*C;
            
            // step 1 : Find the max value over C elements
            float max_val = x_bt[0];
            for(int i = 1; i < C; i++){
                if(x_bt[i] > max_val) max_val = x_bt[i];
            }

            // step 2 : Subtract max, take exp of each, sum them up
            float sum = 0;
            float* out_bt = out + b*T*C + t*C;

            for(int i = 0; i < C; i++){
                sum += exp(x_bt[i] - max_val);
                out_bt[i] = exp(x_bt[i] - max_val);
            }

            // step 3 : Divide each by the sum
            for(int i = 0; i<C; i++){
                out_bt[i] = out_bt[i] / sum;
            }
        }
    }

}



int main(){

    float A[] = {1,2,3,4};  // flat 1D array, shape (3,3)
    float B[] = {1,2,3,4};
    float bias[] = {1,2,3,4};  // bias is 1D, size N
    float out[4] = {0}; 
    float weights[] = {1,1,1,1};
    float eps = 1e-5;

    softmax(out, A, weights, bias, eps, 1,1,4);

    for(int i = 0; i<4; i++){
        cout << out[i] << endl;
    }
    // the output for avove loop : 
    // 0.0320586
    // 0.0871443
    // 0.236883
    // 0.643914
    return 0;
}