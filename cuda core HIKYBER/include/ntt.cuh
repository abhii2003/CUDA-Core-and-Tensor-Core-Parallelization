#ifndef NTT_CUH
#define NTT_CUH

#include "params.h"
#include "reduce.cuh"

// Device-side twiddle factor tables (copied from host at init)
__constant__ int16_t d_zetas[128];
__constant__ int16_t d_zetas_inv[128];

// CT butterfly: (a, b) -> (a + b*zeta, a - b*zeta)
__device__ __forceinline__ void ct_butterfly(int16_t &a, int16_t &b, int16_t zeta)
{
    int16_t t = fqmul(b, zeta);
    b = a - t;
    a = a + t;
}

// GS butterfly: (a, b) -> (a + b, (a - b)*zeta)
__device__ __forceinline__ void gs_butterfly(int16_t &a, int16_t &b, int16_t zeta)
{
    int16_t t = a;
    a = t + b;
    b = fqmul(t - b, zeta);
}

// ---------- EDFS-NTT (Algorithm 6 from paper) ----------
// Entire depth-first search NTT with layer merging
// Sub-layers: 1+6 (paper's best for EDFS on Volta, good starting point for RTX 3050)
// All 256 coefficients loaded upfront, DFS halves data per subtree
__device__ void ntt_edfs(int16_t r[KYBER_N])
{
    int k = 1;
    int len, start, j;

    // Layer 1 (len=128): single full pass — root of the NTT tree
    for (j = 0; j < 128; j++)
    {
        ct_butterfly(r[j], r[j + 128], d_zetas[k]);
    }
    k++;

    // Layers 2–7 via DFS traversal of the remaining 6 layers
    // After layer 1, we have two halves [0..127] and [128..255]
    // DFS each half independently — this is the "entire" DFS
    // Each half is a complete 7-layer sub-NTT on 128 points

    // DFS the full tree of layers 2-7
    // We process by recursively splitting: for each node, do butterfly,
    // then go left, then right (preorder traversal)
    // Iterative implementation using the DFS pattern from Algorithm 6

    // Process left half [0..127] then right half [128..255]
    for (int half = 0; half < 2; half++)
    {
        int base = half * 128;
        int k2 = k + half; // zeta index for this half's layer-2

        // Layer 2 (len=64)
        for (j = 0; j < 64; j++)
            ct_butterfly(r[base + j], r[base + j + 64], d_zetas[k2]);

        // DFS: left subtree of layer 2, then right subtree
        for (int q2 = 0; q2 < 2; q2++)
        {
            int b2 = base + q2 * 64;
            int k3 = 2 * k2 + q2;

            // Layer 3 (len=32)
            for (j = 0; j < 32; j++)
                ct_butterfly(r[b2 + j], r[b2 + j + 32], d_zetas[k3]);

            for (int q3 = 0; q3 < 2; q3++)
            {
                int b3 = b2 + q3 * 32;
                int k4 = 2 * k3 + q3;

                // Layer 4 (len=16)
                for (j = 0; j < 16; j++)
                    ct_butterfly(r[b3 + j], r[b3 + j + 16], d_zetas[k4]);

                for (int q4 = 0; q4 < 2; q4++)
                {
                    int b4 = b3 + q4 * 16;
                    int k5 = 2 * k4 + q4;

                    // Layer 5 (len=8)
                    for (j = 0; j < 8; j++)
                        ct_butterfly(r[b4 + j], r[b4 + j + 8], d_zetas[k5]);

                    for (int q5 = 0; q5 < 2; q5++)
                    {
                        int b5 = b4 + q5 * 8;
                        int k6 = 2 * k5 + q5;

                        // Layer 6 (len=4)
                        for (j = 0; j < 4; j++)
                            ct_butterfly(r[b5 + j], r[b5 + j + 4], d_zetas[k6]);

                        for (int q6 = 0; q6 < 2; q6++)
                        {
                            int b6 = b5 + q6 * 4;
                            int k7 = 2 * k6 + q6;

                            // Layer 7 (len=2) — leaf
                            ct_butterfly(r[b6], r[b6 + 2], d_zetas[k7]);
                            ct_butterfly(r[b6 + 1], r[b6 + 3], d_zetas[k7]);
                        }
                    }
                }
            }
        }
    }

#ifndef USE_PLANTARD
    // Native suite: must reduce all coefficients after NTT (Section III-F)
    for (j = 0; j < KYBER_N; j++)
        r[j] = barrett_reduce(r[j]);
#endif
}

// ---------- SLM-INTT (Algorithm 4, GS butterfly) ----------
// Sub-layers: 1+3+3 (paper's recommendation for INTT)
// GS butterfly: distance goes 2->4->8->...->128
__device__ void intt_slm(int16_t r[KYBER_N])
{
    int k = 127; // start from end of inv zeta table
    int len, start, j;

    // Sub-layer 1: layer 1 only (len=2)
    for (start = 0; start < KYBER_N; start += 4)
    {
        gs_butterfly(r[start], r[start + 2], d_zetas_inv[k]);
        gs_butterfly(r[start + 1], r[start + 3], d_zetas_inv[k]);
        k--;
    }

#ifndef USE_PLANTARD
    // Native suite: Barrett reduction after layer 1 (coeffs grow to 8q)
    for (j = 0; j < KYBER_N; j++)
        r[j] = barrett_reduce(r[j]);
#else
    // Plantard suite: reduction after layer 1 as well
    for (j = 0; j < KYBER_N; j++)
        r[j] = reduce_short(r[j]);
#endif

    // Sub-layer 2: layers 2,3,4 merged (len=4,8,16)
    // Slice N into groups; last layer len=16, so 256/16 = 16 groups of 16
    for (int grp = 0; grp < KYBER_N; grp += 32)
    {
        // len=4
        int16_t z4 = d_zetas_inv[k--];
        for (j = grp; j < grp + 4; j++)
        {
            gs_butterfly(r[j], r[j + 4], z4);
            gs_butterfly(r[j + 16], r[j + 20], z4);
        }
        // len=8
        int16_t z8a = d_zetas_inv[k--];
        int16_t z8b = d_zetas_inv[k--];
        for (j = grp; j < grp + 8; j++)
            gs_butterfly(r[j], r[j + 8], z8a);
        for (j = grp + 16; j < grp + 24; j++)
            gs_butterfly(r[j], r[j + 8], z8b);
        // len=16
        int16_t z16 = d_zetas_inv[k--];
        for (j = grp; j < grp + 16; j++)
            gs_butterfly(r[j], r[j + 16], z16);
    }

    // Reduction after sub-layer 2 (4th layer overall, per paper Section III-F)
    for (j = 0; j < KYBER_N; j++)
        r[j] = barrett_reduce(r[j]);

    // Sub-layer 3: layers 5,6,7 merged (len=32,64,128)
    // len=32
    {
        int16_t z32a = d_zetas_inv[k--];
        int16_t z32b = d_zetas_inv[k--];
        for (j = 0; j < 32; j++)
            gs_butterfly(r[j], r[j + 32], z32a);
        for (j = 64; j < 96; j++)
            gs_butterfly(r[j], r[j + 32], z32a);
        for (j = 128; j < 160; j++)
            gs_butterfly(r[j], r[j + 32], z32b);
        for (j = 192; j < 224; j++)
            gs_butterfly(r[j], r[j + 32], z32b);
    }
    // len=64
    {
        int16_t z64 = d_zetas_inv[k--];
        for (j = 0; j < 64; j++)
            gs_butterfly(r[j], r[j + 64], z64);
        int16_t z64b = d_zetas_inv[k--];
        for (j = 128; j < 192; j++)
            gs_butterfly(r[j], r[j + 64], z64b);
    }
    // len=128
    {
        int16_t z128 = d_zetas_inv[k--];
        for (j = 0; j < 128; j++)
            gs_butterfly(r[j], r[j + 128], z128);
    }

    // Multiply all by n^{-1}
    for (j = 0; j < KYBER_N; j++)
        r[j] = fqmul(r[j], (int16_t)KYBER_NINV_MONT);
}

#endif // NTT_CUH
