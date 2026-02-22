#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include "ntt_table"
#include "intt_table"
#include "test5_p1"
#include "test5_p2"

#define Q 3329
#define N 256
#define R 65536
#define R_INV 169
#define N_INV 3316

__constant__ uint16_t d_ntt_table[128][128];
__constant__ uint16_t d_intt_table[128][128];

using namespace nvcuda::wmma;

__device__ __forceinline__ uint16_t montgomery_reduce(uint32_t T) {
    const uint32_t q_inv = 3327;
    uint32_t m = (T * q_inv) & 0xFFFF;
    uint32_t t = (T + m * Q) >> 16;
    if (t >= Q) t -= Q;
    return (uint16_t)t;
}

__device__ __forceinline__ uint16_t barrett_reduce(uint16_t a) {
    const uint32_t v = 20159;
    uint32_t t = ((uint32_t)v * a) >> 26;
    t = a - t * Q;
    if (t >= Q) t -= Q;
    return (uint16_t)t;
}

// Simplified NTT using INT8 Tensor Cores with proper multi-precision
__global__ void ntt_kernel(const uint16_t* input, uint16_t* output) {
    const int tid = threadIdx.x;
    
    __shared__ uint16_t sh_coeffs[256];
    __shared__ int32_t sh_ntt_result[256];
    
    // Load input
    for (int i = tid; i < 256; i += blockDim.x) {
        sh_coeffs[i] = input[i];
    }
    __syncthreads();
    
    // Compute NTT for even and odd coefficients separately
    // Even: f̂_{2i} = Σ_j f_{2j} · ζ^{(2br7(i)+1)j}
    // Odd:  f̂_{2i+1} = Σ_j f_{2j+1} · ζ^{(2br7(i)+1)j}
    
    for (int output_idx = tid; output_idx < 256; output_idx += blockDim.x) {
        int is_odd = output_idx % 2;
        int i = output_idx / 2;  // Which NTT output (0-127)
        
        int64_t sum = 0;
        
        // Compute dot product
        for (int j = 0; j < 128; j++) {
            uint32_t coeff = sh_coeffs[2 * j + is_odd];  // f_{2j} or f_{2j+1}
            uint32_t twiddle = d_ntt_table[i][j];
            
            // Split into 6-bit parts for multi-precision
            uint32_t c_low = coeff & 0x3F;
            uint32_t c_high = (coeff >> 6) & 0x3F;
            uint32_t t_low = twiddle & 0x3F;
            uint32_t t_high = (twiddle >> 6) & 0x3F;
            
            // Full product = (c_low + c_high*64) * (t_low + t_high*64)
            //              = c_low*t_low + (c_low*t_high + c_high*t_low)*64 + c_high*t_high*4096
            uint64_t prod = c_low * t_low;
            prod += (c_low * t_high + c_high * t_low) * 64;
            prod += c_high * t_high * 4096;
            
            sum += prod;
        }
        
        // Apply modular reduction
			sh_ntt_result[output_idx] = montgomery_reduce((uint32_t)(sum % (uint64_t)(Q*R)));
    }
    
    __syncthreads();
    
    // Write output
    for (int i = tid; i < 256; i += blockDim.x) {
        output[i] = (uint16_t)sh_ntt_result[i];
    }
}

// INTT kernel (similar structure)
__global__ void intt_kernel(const uint16_t* input, uint16_t* output) {
    const int tid = threadIdx.x;
    
    __shared__ uint16_t sh_coeffs[256];
    __shared__ int32_t sh_result[256];
    
    // Load NTT coefficients
    for (int i = tid; i < 256; i += blockDim.x) {
        sh_coeffs[i] = input[i];
    }
    __syncthreads();
    
    // Compute INTT
    // f_{2j} = n^{-1} · Σ_{i=0}^{127} f̂_{2i} · ζ^{-(2·br7(i)+1)·j}
    // f_{2j+1} = n^{-1} · Σ_{i=0}^{127} f̂_{2i+1} · ζ^{-(2·br7(i)+1)·j}
    
    for (int output_idx = tid; output_idx < 256; output_idx += blockDim.x) {
        int is_odd_output = output_idx & 1;  // Output is even or odd coefficient
        int j_out = output_idx >> 1;         // Output position (0-127)
        
        int64_t sum = 0;
        
        // Sum over all 128 NTT coefficients
        for (int i_sum = 0; i_sum < 128; i_sum++) {
            // Read NTT coefficient f̂_{2i} or f̂_{2i+1}
            uint32_t ntt_coeff = sh_coeffs[2 * i_sum + is_odd_output];
            
            // Get inverse twiddle: n^{-1} · ζ^{-(2·br7(i_sum)+1)·j_out} · R
            // CRITICAL: indices are [summation_idx][output_idx]
            uint32_t twiddle = d_intt_table[i_sum][j_out];
            
            // Multi-precision multiplication
            uint32_t c_low = ntt_coeff & 0x3F;
            uint32_t c_high = (ntt_coeff >> 6) & 0x3F;
            uint32_t t_low = twiddle & 0x3F;
            uint32_t t_high = (twiddle >> 6) & 0x3F;
            
            uint64_t prod = c_low * t_low;
            prod += (c_low * t_high + c_high * t_low) * 64;
            prod += c_high * t_high * 4096;
            
            sum += prod;
        }
        
        // Apply Montgomery reduction to convert from Montgomery domain
           sh_result[output_idx] = montgomery_reduce( (uint32_t)(sum % (uint64_t)(Q*R)));
    }
    
    __syncthreads();
    
    // Write output
    for (int i = tid; i < 256; i += blockDim.x) {
        output[i] = (uint16_t)sh_result[i];
    }
}

// Pointwise multiplication (basecase multiplication for Kyber)
__global__ void pointwise_mul_kernel(const uint16_t* f_ntt, const uint16_t* g_ntt, uint16_t* h_ntt) {
    int i = threadIdx.x;
    if (i >= 128) return;
    
    uint32_t f0 = f_ntt[2*i];
    uint32_t f1 = f_ntt[2*i + 1];
    uint32_t g0 = g_ntt[2*i];
    uint32_t g1 = g_ntt[2*i + 1];
    
    // Get zeta (this is in Montgomery form from table)
    uint32_t zeta = d_ntt_table[i][1];
    
    // All inputs are standard form, do regular multiplication mod Q
    // h0 = f0*g0 + f1*g1*zeta (mod Q)
    uint32_t t0 = (f0 * g0) % Q;
    uint32_t t1 = (f1 * g1) % Q;
    // zeta is in Montgomery form, so zeta*t1 needs one montgomery_reduce
    t1 = montgomery_reduce(t1 * zeta);  // Only here!
    uint32_t h0 = (t0 + t1) % Q;
    
    // h1 = f0*g1 + f1*g0 (mod Q)
    uint32_t t2 = (f0 * g1) % Q;
    uint32_t t3 = (f1 * g0) % Q;
    uint32_t h1 = (t2 + t3) % Q;
    
    h_ntt[2*i] = (uint16_t)h0;
    h_ntt[2*i + 1] = (uint16_t)h1;
}

// Normalize by multiplying with N_INV
__global__ void normalize_kernel(uint16_t* data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 256) {
        data[i] = ((int32_t)data[i]) % Q;
    }
}

int main() {
	//NOTE: THIS IS THE INTIALIZATION PHASE, MOSTLY ALLOCATIONS
    cudaMemcpyToSymbol(d_ntt_table, ntt_table, sizeof(uint16_t) * 128 * 128);
    cudaMemcpyToSymbol(d_intt_table, intt_table, sizeof(uint16_t) * 128 * 128);
    
    uint16_t result[N];
    
    uint16_t *d_p1, *d_p2, *d_p1_ntt, *d_p2_ntt, *d_h_ntt, *d_result;
    cudaMalloc(&d_p1, sizeof(uint16_t) * N);
    cudaMalloc(&d_p2, sizeof(uint16_t) * N);
    cudaMalloc(&d_p1_ntt, sizeof(uint16_t) * N);
    cudaMalloc(&d_p2_ntt, sizeof(uint16_t) * N);
    cudaMalloc(&d_h_ntt, sizeof(uint16_t) * N);
    cudaMalloc(&d_result, sizeof(uint16_t) * N);
    
    cudaMemcpy(d_p1, p1, sizeof(uint16_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, p2, sizeof(uint16_t) * N, cudaMemcpyHostToDevice);
   //=========================================================================

   //NOTE: VARIOUS DEBUG STATEMENTS AFTER AND BEFORE EACH OPEARTION
   // Check p1 input
   uint16_t debug_p1_input[16];
   cudaMemcpy(debug_p1_input, d_p1, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
   printf("p1 input: ");
   for(int i = 0; i < 16; i++) printf("%u ", debug_p1_input[i]);
   printf("\n");
   
   ntt_kernel<<<1, 256>>>(d_p1, d_p1_ntt);
   cudaDeviceSynchronize();
   
   uint16_t debug_ntt[16];
   cudaMemcpy(debug_ntt, d_p1_ntt, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
   printf("After NTT(p1): ");
   for(int i = 0; i < 16; i++) printf("%u ", debug_ntt[i]);
   printf("\n");
   //================================P2 CHECK================================= 
   // Check p2 input
   uint16_t debug_p2_input[16];
   cudaMemcpy(debug_p2_input, d_p2, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
   printf("p2 input: ");
   for(int i = 0; i < 16; i++) printf("%u ", debug_p2_input[i]);
   printf("\n");
   
   ntt_kernel<<<1, 256>>>(d_p2, d_p2_ntt);
   cudaDeviceSynchronize();
   
   uint16_t debug_ntt2[16];
   cudaMemcpy(debug_ntt2, d_p2_ntt, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
   printf("After NTT(p2): ");
   for(int i = 0; i < 16; i++) printf("%u ", debug_ntt2[i]);
   printf("\n");

    // Pointwise multiplication
    pointwise_mul_kernel<<<1, 128>>>(d_p1_ntt, d_p2_ntt, d_h_ntt);
    uint16_t debug_ntt_m[16];
            cudaMemcpy(debug_ntt_m, d_h_ntt, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
            printf("After pointwise:        ");
            for(int i = 0; i < 16; i++) printf("%u ", debug_ntt_m[i]);
            printf("\n");
    cudaDeviceSynchronize();
    
    // Inverse NTT
    intt_kernel<<<1, 256>>>(d_h_ntt, d_result);
	cudaDeviceSynchronize();

    //cudaDeviceSynchronize();
    //**DEBUG STEP FOR INTT
    uint16_t debug_intt[16];
                cudaMemcpy(debug_intt, d_result, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                printf("After INTT:             ");
                for(int i = 0; i < 16; i++) printf("%u ", debug_intt[i]);
                printf("\n");
    cudaDeviceSynchronize();
   //after normalization
    normalize_kernel<<<1, 256>>>(d_result);
	uint16_t debug_norm[16];
				cudaMemcpy(debug_norm, d_result, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				printf("After Norm:             ");
				for(int i = 0; i < 16; i++) printf("%u ", debug_norm[i]);
				printf("\n");
	cudaDeviceSynchronize();		
    
    // Copy result back
    cudaMemcpy(result, d_result, sizeof(uint16_t) * N, cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", result[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    
    
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_p1_ntt);
    cudaFree(d_p2_ntt);
    cudaFree(d_h_ntt);
    cudaFree(d_result);
    
    return 0;
}
