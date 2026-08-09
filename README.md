# cpp-gpu-inference

This repo covers end to end cpp-gpu-inference.

Topics that I have covered so far:

1. Cpp
2. Coding Transformer from Scratch in Cpp
3. Cuda
   - This includes learning cuda
   - PMPP book was used for this
   - Basic cuda covered
   - profiling intro covered
   - optimizing the matrix multiplication by these techniques : 
      - profiling
      - tiling
      - thread coarsening
      - Corner Turning
4. Triton Kernels
   - matrix multilication
   - vector addition
   - softmax 
   - sasha rush gpu puzzles
5. Flash Attention
   - Forward FA handtrace
   - Backward FA handtrace
   - Implementing both Forward and backward FA kernels
   - FA-2 Implementation has been used
   - Comparing FA-1 VS FA-2
   - Finally, using torch Autograd function to make the kernels compatible in order to be able to port to custom transformer

6. Quantization
   - LLM.int8()
   - SmoothQuant
   - GPTQ
   - AWQ

7. 