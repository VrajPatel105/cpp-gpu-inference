#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;

// Transformer blocks individually.

// 1. Embeddings
void embeddings_forward(float* out, int* tokens, float* weight, int B, int T, int d_model){
    // A lookup table. Given a token ID, fetch its row from the weight matrix. Then scale by sqrt(d_model)
    // steps :  
    // 1. Loop over each token in the batch
    float scale_factor = sqrt(d_model);
    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){
            int curr_token = tokens[b*T + t];
            // 2. Look up its row in the weight matrix
            for(int row = 0; row < d_model; row++){
                // 3. Scale each value by sqrt(d_model)
                // 4. Write to output
                out[b*T*d_model + t*d_model + row] = weight[curr_token*d_model + row] * scale_factor;
            }
        }
    }
}



// 2. Positional Encoding
void positional_encoding(float* x, int B, int T, int d_model){
    // Transformers have no sense of order by default. 
    // PE injects position information by adding a fixed vector to each token's embedding. 
    // The values come from sin/cos formulas, not learned.

    // Formulas used : 
    // PE[pos][2i]   = sin(pos / 10000^(2i/d_model))
    // PE[pos][2i+1] = cos(pos / 10000^(2i/d_model))

    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){
            for(int i = 0; i<d_model/2; i++){
                float den = pow(10000,(2.0f*i / d_model));
                float even = sin(t/den);
                float odd = cos(t/den);
                x[b*T*d_model + t*d_model + 2*i] += even;
                x[b*T*d_model + t*d_model + 2*i+1] += odd;
            }
        }
    }
}


// function for matrix multiplication
// @param A - input 
// @param B - weight 
// @param out - output
void matmul(float* A, float* B, float* bias, float* out, int M, int K, int N){
    for(int m = 0; m<M; m++){
        for(int n = 0; n<N; n++){
            float val = (bias != nullptr) ? bias[n] : 0.0f;
            for(int k = 0; k<K; k++){
                val += A[m*K + k] * B[k*N + n];
            }
            out[m*N + n] = val;
        }
    }
}


// feed forward network
void feedforward_forward(float* out, float* x, float* W1, float* b1, float* W2, float* b2, int B, int T, int d_model, int d_ff){
    float* intermediate = new float[B * T * d_ff]();
    //new float[size] asks the OS for a chunk of memory big enough to hold size floats and returns a pointer to the start. 
    // The () zero-initializes it.

    // First matmul : x * W1 -> intermediate
    matmul(x, W1, b1, intermediate, B*T, d_model, d_ff);

    // relu operation -> loop over every element in the intermediate/buffer and if the value is > 0, make the value 0.
    for(int ele = 0; ele < (B*T*d_ff); ele++){
        if(intermediate[ele] < 0) intermediate[ele] = 0;
    }

    // Second matmul : intermediate * W2 -> out
    matmul(intermediate, W2, b2, out, B*T, d_ff, d_model);

    delete[] intermediate; // free up the mem
}


// writing output matrix printing func. -> this is better in terms of viz. prints 2D matrix
void PrintOutputMatrix(float* weight, float* arr){
    cout << "\nPrinting the weight matrix" << endl;
    for(int i = 0; i<16; i++){
        if(i % 4 == 0) cout << "\n";
        cout << weight[i] << "  ";
    }
    cout<< "\n\nPrinting the output matrix" << endl;
    for(int i = 0; i<16; i++){
        if(i % 4 == 0) cout << "\n";
        cout << arr[i] << "  ";
    }
}

// This function below prints flatten array
void PrintOutputFlat(float* weight, float* arr){
    cout << "\nPrinting the weight - output matrix " << endl;
    for(int i = 0; i<16; i++){
        cout << weight[i] << " - " << arr[i] << endl;
    }
}


int main(){

    // basic variables / lists required to call the functions 
    int B = 1;
    int T = 4;
    int d_model = 4;
    int d_ff = 16;
    int tokens[] = {2, 0, 3, 1};
    float weight[] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; 
    float out[16] = {0};
    float W1[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
    float b1[16] = {0};
    float W2[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    float b2[4] = {0}; 
    float ff_out[16] = {0};

    embeddings_forward(out, tokens, weight, B, T, d_model);

    positional_encoding(out,B,T,d_model); // here out is basically x which is the input.

    feedforward_forward(ff_out, out, W1, b1, W2, b2, B, T, d_model, d_ff);

    PrintOutputMatrix(weight, ff_out);
    cout << "\n\n";
    PrintOutputFlat(weight, ff_out);

    return 0;
}