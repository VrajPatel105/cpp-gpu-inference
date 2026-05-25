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
    int tokens[] = {2, 0, 3, 1};
    float weight[] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; 
    float out[16] = {0}; 

    embeddings_forward(out, tokens, weight, B, T, d_model);

    PrintOutputMatrix(weight, out);
    cout << "\n\n";
    PrintOutputFlat(weight, out);

    return 0;
}