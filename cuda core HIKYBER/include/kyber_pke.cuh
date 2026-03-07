#ifndef KYBER_PKE_CUH
#define KYBER_PKE_CUH

#include "params.h"
#include "poly.cuh"
#include "kyber_utils.cuh"

// ========== Fused CPA-PKE Kernels ==========
// One thread = one complete Kyber instance
// All sub-operations fused: only 1 global load + 1 global store per kernel
// (Table I from paper: reduces global memory accesses from 12/17/8 to 1/1/1)

// ---------- KeyGen ----------
// Input:  d[32] random bytes per instance
// Output: pk[KYBER_INDCPA_PUBLICKEYBYTES], sk[KYBER_INDCPA_SECRETKEYBYTES]
__global__ void pke_keygen_kernel(const uint8_t *d_in,
                                  uint8_t *pk_out,
                                  uint8_t *sk_out,
                                  int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *d = d_in + tid * KYBER_SYMBYTES;
    uint8_t *pk = pk_out + tid * KYBER_INDCPA_PUBLICKEYBYTES;
    uint8_t *sk = sk_out + tid * KYBER_INDCPA_SECRETKEYBYTES;

    // Step 1: (rho, sigma) = SHA3-512(d)
    uint8_t buf[64];
    sha3_512(buf, d, KYBER_SYMBYTES);
    uint8_t rho[KYBER_SYMBYTES], sigma[KYBER_SYMBYTES];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
    {
        rho[i] = buf[i];
        sigma[i] = buf[KYBER_SYMBYTES + i];
    }

    // Step 2: A_hat = Gen_matrix(rho) in NTT domain
    polyvec a_hat[KYBER_K];
    gen_matrix(a_hat, rho, 0);

    // Step 3: (s, e) = CBD_sample(sigma)
    polyvec s, e;
    uint8_t nonce = 0;
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&s.vec[i], sigma, nonce++);
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&e.vec[i], sigma, nonce++);

    // Step 4: (s_hat, e_hat) = (NTT(s), NTT(e))
    polyvec_ntt(&s);
    polyvec_ntt(&e);

    // Step 5: t_hat = A_hat ◦ s_hat + e_hat
    polyvec t_hat;
    for (int i = 0; i < KYBER_K; i++)
    {
        polyvec_basemul_acc(&t_hat.vec[i], &a_hat[i], &s);
        poly_add(&t_hat.vec[i], &t_hat.vec[i], &e.vec[i]);
    }
    polyvec_reduce(&t_hat);

    // Step 6: pk = Encode(t_hat || rho), sk = Encode(s_hat)
    pack_pk(pk, &t_hat, rho);
    pack_sk(sk, &s);
}

// ---------- Encryption ----------
// Input:  pk, msg[32], coins[32]
// Output: ct[KYBER_INDCPA_BYTES]
__global__ void pke_enc_kernel(const uint8_t *pk_in,
                               const uint8_t *msg_in,
                               const uint8_t *coins_in,
                               uint8_t *ct_out,
                               int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *pk = pk_in + tid * KYBER_INDCPA_PUBLICKEYBYTES;
    const uint8_t *msg = msg_in + tid * KYBER_SYMBYTES;
    const uint8_t *coins = coins_in + tid * KYBER_SYMBYTES;
    uint8_t *ct = ct_out + tid * KYBER_INDCPA_BYTES;

    // Step 1: (t_hat, rho) = Decode(pk)
    polyvec t_hat;
    uint8_t rho[KYBER_SYMBYTES];
    unpack_pk(&t_hat, rho, pk);

    // Step 2: A_hat^T = Gen_matrix(rho) transposed
    polyvec a_hat[KYBER_K];
    gen_matrix(a_hat, rho, 1); // transposed

    // Step 3: Sample r, e1, e2
    polyvec r_vec;
    polyvec e1;
    poly e2;
    uint8_t nonce = 0;
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&r_vec.vec[i], coins, nonce++);
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta2(&e1.vec[i], coins, nonce++);
    sample_poly_cbd_eta2(&e2, coins, nonce++);

    // Step 4: r_hat = NTT(r)
    polyvec_ntt(&r_vec);

    // Step 5: u = INTT(A_hat^T ◦ r_hat) + e1
    polyvec u;
    for (int i = 0; i < KYBER_K; i++)
    {
        polyvec_basemul_acc(&u.vec[i], &a_hat[i], &r_vec);
    }
    polyvec_intt(&u);
    polyvec_add(&u, &u, &e1);
    polyvec_reduce(&u);

    // Step 6: v = INTT(t_hat^T ◦ r_hat) + e2 + Decompress(m)
    poly v;
    polyvec_basemul_acc(&v, &t_hat, &r_vec); // t_hat is already a single polyvec
    // Actually t_hat^T ◦ r_hat is inner product:
    // Already handled by polyvec_basemul_acc which does sum of basemul
    poly_intt(&v);
    poly_add(&v, &v, &e2);

    poly mp;
    poly_frommsg(&mp, msg);
    poly_add(&v, &v, &mp);
    poly_reduce(&v);

    // Step 7: c = (Encode(Compress(u)), Encode(Compress(v)))
    polyvec_csubq(&u);
    poly_csubq(&v);
    pack_ciphertext(ct, &u, &v);
}

// ---------- Decryption ----------
// Input:  ct[KYBER_INDCPA_BYTES], sk[KYBER_INDCPA_SECRETKEYBYTES]
// Output: msg[32]
__global__ void pke_dec_kernel(const uint8_t *ct_in,
                               const uint8_t *sk_in,
                               uint8_t *msg_out,
                               int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *ct = ct_in + tid * KYBER_INDCPA_BYTES;
    const uint8_t *sk = sk_in + tid * KYBER_INDCPA_SECRETKEYBYTES;
    uint8_t *msg = msg_out + tid * KYBER_SYMBYTES;

    // Step 1: (u, v) = Decode(c)
    polyvec u;
    poly v;
    unpack_ciphertext(&u, &v, ct);

    // Step 2: s_hat = Decode(sk)
    polyvec s_hat;
    unpack_sk(&s_hat, sk);

    // Step 3: m = Compress(v - INTT(s_hat ◦ NTT(u)))
    polyvec_ntt(&u);

    poly tmp;
    polyvec_basemul_acc(&tmp, &s_hat, &u);
    poly_intt(&tmp);

    poly_sub(&v, &v, &tmp);
    poly_reduce(&v);
    poly_csubq(&v);

    // Step 4: Encode message
    poly_tomsg(msg, &v);
}

#endif // KYBER_PKE_CUH
