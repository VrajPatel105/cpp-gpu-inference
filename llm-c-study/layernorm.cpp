#include <iostream>
#include <math.h>
using namespace std;

void layernorm(float* out, float* x, float* weight, float* bias, float eps, int B, int T, int C){
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
            // step 1 : calculate the mean
            float total_sum = 0;
            for(int i = 0; i<C; i++){
                total_sum += x_bt[i];
            }
            float mean = total_sum / C;

            // step 2 : calculate the variance
            float sum = 0;
            for(int i = 0; i<C; i++){
                sum += (x_bt[i] - mean)*(x_bt[i] - mean);
            }
            float var = sum / C;
            
            // step 3 : calculate the rstd
            float rstd = 1 / sqrt(var + eps);

            // step 4 : Normalize + Scale + shift
            float* out_bt = out + b*T*C + t*C;
            for(int i = 0; i < C; i++){
                out_bt[i] = (x_bt[i] - mean) * rstd * weight[i] + bias[i];
            }
        }
    }

}


// The four steps on one C-dim vector:

// Mean: average of all C elements. mean = (x[0] + x[1] + ... + x[C-1]) / C
// Variance: average squared deviation from mean. var = sum of (x[i] - mean)^2 / C
// rstd: reciprocal standard deviation. rstd = 1 / sqrt(var + eps). The eps (usually 1e-5) prevents divide by zero.
// Normalize + scale + shift: out[i] = (x[i] - mean) * rstd * weight[i] + bias[i]

// The loop structure:
// Outer loop over every token position (B × T positions total). For each position, you have one C-dim vector. 
// Run the four steps on that vector. Write C output values.
// Flat indexing for this function:
// To get to token (b, t)'s vector in x: x + b*T*C + t*C

int main(){
    float A[] = {1,2,3,4};  // flat 1D array, shape (3,3)
    float B[] = {1,2,3,4};
    float bias[] = {1,2,3,4};  // bias is 1D, size N
    float out[4] = {0}; 
    float weights[] = {1,1,1,1};
    float eps = 1e-5;

    layernorm(out, A, weights, bias, eps, 1,1,4);

    for(int i = 0; i<4; i++){
        cout << out[i] << endl;
    }

    return 0;
}