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
// @param A input 
// @param B weight 
// @param out output
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

void softmax(float* out, float* x, int B, int T, int C){
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

// MHA function
void attention_forward(float* out, float* x, float* k_input, float* v_input,
                       float* Wq, float* Wk, float* Wv, float* Wo,
                       int B, int T, int num_heads, int d_model, bool causal){
    // The steps to be followed: 
    // 1. Multiply x with weight matrices of Q,K,V : x * Wq -> x * Wk -> x * Wv 
    // 2. split into num heads and also initialize the d_k var
    // 3. implement the formula : Attention(Q,K,V) = softmax((Q*K.transpose)/root(d_k)) * V
    // 4. write to output

    // first we need three bufferes since it's not a good practice to dirctly modify input 'x'.
    float* Q = new float[B * T * d_model]();
    float* K = new float[B * T * d_model]();
    float* V = new float[B * T * d_model]();
    float* scores = new float[B * num_heads * T * T]();
    float* attn_weights = new float[B * num_heads * T * T]();
    float* attn_out = new float[B * T * d_model]();

    // multiply the Q,K,V with their corresponding weights Wq, Wk, Wv
    matmul(x, Wq, nullptr, Q, B*T, d_model, d_model);
    matmul(k_input, Wk, nullptr, K, B*T, d_model, d_model);
    matmul(v_input, Wv, nullptr, V, B*T, d_model, d_model);
    
    // split into heads
    int d_k = d_model / num_heads; // we should also add a checker here that checks that this is divisible. 

    // we actually dont have to split into heads since Q,K and V are actuall 1D arrays (matrix are just for our viz). 
    // so accordintg to the patter, we will use : b*T*num_heads*d_k + t*num_heads*d_k + h*d_k + i
    
    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){
            for(int h = 0; h<num_heads; h++){
                for(int t2 = 0; t2<T; t2++){
                    float val = 0;
                    for(int i = 0; i<d_k; i++){
                        val += Q[b*T*num_heads*d_k + t*num_heads*d_k + h*d_k + i] * K[b*T*num_heads*d_k + t2*num_heads*d_k + h*d_k + i];
                    }  
                    scores[b*num_heads*T*T + h*T*T + t*T + t2] = val;
                    scores[b*num_heads*T*T + h*T*T + t*T + t2] /= sqrt(d_k);
                    if(t2 > t) scores[b*num_heads*T*T + h*T*T + t*T + t2] = -1e9f;
                }

                // softmax : 
                // initializing few vars
                float* score_slice = scores + b*num_heads*T*T + h*T*T + t*T;
                float* attn_slice = attn_weights + b*num_heads*T*T + h*T*T + t*T;
                // step 1 : Find the max value over C elements
                float max_val = score_slice[0];
                for(int i = 1; i < T; i++){
                    if(score_slice[i] > max_val) max_val = score_slice[i];
                }
                // step 2 : Subtract max, take exp of each, sum them up
                float sum = 0;
                for(int i = 0; i < T; i++){
                    sum += exp(score_slice[i] - max_val);
                    attn_slice[i] = exp(score_slice[i] - max_val);
                }
                // step 3 : Divide each by the sum
                for(int i = 0; i<T; i++){
                    attn_slice[i] = attn_slice[i] / sum;
                }

                // now multiply by V
                for(int i = 0; i<d_k; i++){
                    float val = 0;
                    for(int t2 = 0; t2<T; t2++){
                        val += attn_weights[b*num_heads*T*T + h*T*T + t*T + t2] * V[b*T*num_heads*d_k + t2*num_heads*d_k + h*d_k + i];
                    }
                    attn_out[b*T*num_heads*d_k + t*num_heads*d_k + h*d_k + i] = val;
                }
                
            }
        }
    }
    // now finally multiply attn_out with Wo
    matmul(attn_out, Wo, nullptr, out, B*T, d_model, d_model);

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] scores;
    delete[] attn_weights;
    delete[] attn_out;
}


void layernorm(float* out, float* x, float* gamma, float* beta, float eps, int B, int T, int d_model){

    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            // lets also get the pointer the the token's C-dim vector
            float* x_bt = x + b*T*d_model + t*d_model;
            // step 1 : calculate the mean
            float total_sum = 0;
            for(int i = 0; i<d_model; i++){
                total_sum += x_bt[i];
            }
            float mean = total_sum / d_model;

            // step 2 : calculate the variance
            float sum = 0;
            for(int i = 0; i<d_model; i++){
                sum += (x_bt[i] - mean)*(x_bt[i] - mean);
            }
            float var = sum / d_model;
            
            // step 3 : calculate the rstd
            float rstd = 1 / sqrt(var + eps);

            // step 4 : Normalize + Scale + shift
            float* out_bt = out + b*T*d_model + t*d_model;
            for(int i = 0; i < d_model; i++){
                out_bt[i] = (x_bt[i] - mean) * rstd * gamma[i] + beta[i];
            }
        }
    }
}

// residual connection block
void residual(float* out, float* x, float* sublayer_out, int B, int T, int d_model){
    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){
            for(int row = 0; row < d_model; row++){
                out[b*T*d_model + t*d_model + row] = x[b*T*d_model + t*d_model + row] + sublayer_out[b*T*d_model + t*d_model + row];
            }
        }
    }
}


// Encoder block

void encoder_block(float* out, float* x,
                   float* Wq, float* Wk, float* Wv, float* Wo,
                   float* W1, float* b1, float* W2, float* b2,
                   float* gamma1, float* beta1,
                   float* gamma2, float* beta2,
                   float eps, int B, int T, int num_heads, int d_model, int d_ff){


                    // initializing the buffers that we need beforehand
                    float* attn_out = new float[B * T * d_model];
                    float* residual1 = new float[B * T * d_model];
                    float* norm1 = new float[B * T * d_model];
                    float* ff_out = new float[B * T * d_model];
                    float* residual2 = new float[B * T * d_model];

                    // now applying the functions
                    // 1. x = layernorm(x + attention(x))
                    // 2. x = layernorm(x + feed_forward(x))

                    attention_forward(attn_out, x, x, x, Wq, Wk, Wv, Wo, B, T, num_heads, d_model, false);
                    residual(residual1, x, attn_out, B, T, d_model);
                    layernorm(norm1, residual1, gamma1, beta1, eps, B, T, d_model);
                    feedforward_forward(ff_out, norm1, W1, b1, W2, b2, B, T, d_model, d_ff);
                    residual(residual2, norm1, ff_out, B, T, d_model);
                    layernorm(out, residual2, gamma2, beta2, eps, B, T, d_model);


                    // free up the initialized buffers
                    delete[] attn_out;
                    delete[] residual1;
                    delete[] norm1;
                    delete[] ff_out;
                    delete[] residual2; 
                   }



// Decoder block 
void decoder_block(float* out, float* x, float* enc_out,
                   float* Wq1, float* Wk1, float* Wv1, float* Wo1,  // masked self-attn
                   float* Wq2, float* Wk2, float* Wv2, float* Wo2,  // cross-attn
                   float* W1, float* b1, float* W2, float* b2,
                   float* gamma1, float* beta1,
                   float* gamma2, float* beta2,
                   float* gamma3, float* beta3,
                   float eps, int B, int T, int num_heads, int d_model, int d_ff){

                    // the order : 
                    // 1. x = layernorm(x + masked_attention(x))
                    // 2. x = layernorm(x + cross_attention(x, enc_out))
                    // 3. x = layernorm(x + feedforward(x))
                    
                    // lets define the buffers needed.
                    float* mask_attn_out = new float[B * T * d_model];
                    float* residual1 = new float[B * T * d_model];
                    float* norm1 = new float[B * T * d_model];
                    float* cross_attn_out = new float[B * T * d_model];
                    float* residual2 = new float[B * T * d_model];
                    float* norm2 = new float[B * T * d_model];
                    float* ffn_out = new float[B * T * d_model];
                    float* residual3 = new float[B * T * d_model];


                    // calling the functions for decoder

                    // 1. x = layernorm(x + masked_attention(x))
                    attention_forward(mask_attn_out, x, x, x, Wq1, Wk1, Wv1, Wo1, B, T, num_heads, d_model, true);
                    residual(residual1, x, mask_attn_out, B, T, d_model);
                    layernorm(norm1, residual1, gamma1, beta1, eps, B, T, d_model);
                    
                    // 2. x = layernorm(x + cross_attention(x, enc_out))
                    attention_forward(cross_attn_out, norm1, enc_out, enc_out, Wq2, Wk2, Wv2, Wo2, B, T, num_heads, d_model, false); // note that Wk2 and Wv2 are encoder ouputs
                    residual(residual2, norm1, cross_attn_out, B, T, d_model);
                    layernorm(norm2, residual2, gamma2, beta2, eps, B, T, d_model);
                    
                    // 3. x = layernorm(x + feedforward(x))
                    feedforward_forward(ffn_out, norm2, W1, b1, W2, b2, B, T, d_model, d_ff);
                    residual(residual3, norm2, ffn_out, B, T, d_model);
                    layernorm(out, residual3, gamma3, beta3, eps, B, T, d_model);


                    // finally freeing up the memory
                    delete[] mask_attn_out;
                    delete[] residual1;
                    delete[] norm1;
                    delete[] cross_attn_out;
                    delete[] residual2;
                    delete[] norm2;
                    delete[] ffn_out;
                    delete[] residual3;
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
    float pe_out[16] = {0};
    float enc_out[16] = {0};
    float W1[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
    float b1[16] = {0};
    float W2[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    float b2[4] = {0}; 
    float ff_out[16] = {0};
    int num_heads = 2;
    float Wq[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wk[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wv[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wo[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    
    float Wq1[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wk1[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wv1[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wo1[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    float Wq2[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wk2[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wv2[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float Wo2[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    float mha_out[16] = {0};
    float eps = 1e-5;
    float gamma1[4] = {1,1,1,1};
    float beta1[4]  = {0,0,0,0};
    float gamma2[4] = {1,1,1,1};
    float beta2[4]  = {0,0,0,0};
    float gamma3[4] = {1,1,1,1};
    float beta3[4]  = {0,0,0,0};
    float dec_out[16] = {0};


    embeddings_forward(out, tokens, weight, B, T, d_model);

    positional_encoding(out,B,T,d_model); // here out is basically x which is the input.

    // feedforward_forward(ff_out, out, W1, b1, W2, b2, B, T, d_model, d_ff);

    // attention_forward(mha_out, out, Wq, Wk, Wv, Wo, B, T, num_heads, d_model);

    // layernorm();

    // residual();

    encoder_block(enc_out, out, Wq, Wk, Wv, Wo, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, eps, B, T, num_heads, d_model, d_ff);

    // abve function call output : 
    // weight - output matrix 
    // 1 - -1.13126
    // 0 - -0.5201
    // 0 - 1.56264
    // 0 - 0.0887169
    // 0 - 1.59509
    // 1 - -0.727529
    // 0 - -0.958211
    // 0 - 0.0906536
    // 0 - 0.0421487
    // 0 - -0.840558
    // 1 - -0.820005
    // 0 - 1.61841
    // 0 - -0.547029
    // 0 - -0.0884168
    // 0 - -1.00346
    // 1 - 1.6389


    // calling decoder block
    decoder_block(dec_out, out, enc_out, Wq1, Wk1, Wv1, Wo1, Wq2, Wk2, Wv2, Wo2, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, gamma3, beta3, eps, B, T, num_heads, d_model, d_ff);

    // otput for decoder : 
    // weight - output matrix 
    // 1 - -1.07892
    // 0 - -0.124557
    // 0 - 1.62551
    // 0 - -0.422041
    // 0 - 0.817816
    // 1 - -0.522881
    // 0 - -1.37512
    // 0 - 1.08018
    // 0 - -0.218976
    // 0 - -0.341237
    // 1 - -1.07611
    // 0 - 1.63633
    // 0 - -0.707657
    // 0 - 0.251202
    // 0 - -1.06249
    // 1 - 1.51895

    PrintOutputMatrix(weight, dec_out);
    cout << "\n\n";
    PrintOutputFlat(weight, dec_out);

    return 0;
}