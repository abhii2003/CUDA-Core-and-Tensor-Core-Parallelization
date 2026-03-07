#ifndef KYBER_UTILS_CUH
#define KYBER_UTILS_CUH

#include "params.h"
#include "poly.cuh"
#include "keccak.cuh"

// ---- Centered Binomial Distribution (η = 2) ----
// Sample polynomial with coefficients in {-2, -1, 0, 1, 2}
// Input: 128 bytes (for η=2: 2*η*N/8 = 128)
__device__ void cbd_eta2(poly *r, const uint8_t buf[128])
{
    for (int i = 0; i < KYBER_N / 4; i++)
    {
        uint32_t t = ((const uint32_t *)buf)[i];
        uint32_t d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;
        for (int j = 0; j < 4; j++)
        {
            int16_t a = (d >> (8 * j)) & 0x3;
            int16_t b = (d >> (8 * j + 2)) & 0x3;
            r->coeffs[4 * i + j] = a - b;
        }
    }
}

// Alias for eta1 = eta2 = 2 in Kyber-768
__device__ __forceinline__ void cbd_eta1(poly *r, const uint8_t buf[128])
{
    cbd_eta2(r, buf);
}

// ---- Matrix generation (Â from seed ρ via SHAKE-128) ----
// Rejection sampling: parse SHAKE-128 output into uniform mod q
__device__ int rej_uniform(int16_t *r, int len, const uint8_t *buf, int buflen)
{
    int ctr = 0, pos = 0;
    while (ctr < len && pos + 3 <= buflen)
    {
        uint16_t val0 = ((uint16_t)buf[pos] | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF;
        uint16_t val1 = (((uint16_t)buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF;
        pos += 3;
        if (val0 < KYBER_Q)
            r[ctr++] = (int16_t)val0;
        if (ctr < len && val1 < KYBER_Q)
            r[ctr++] = (int16_t)val1;
    }
    return ctr;
}

// Generate single polynomial of matrix Â[i][j] from seed
__device__ void gen_matrix_entry(poly *entry, const uint8_t seed[KYBER_SYMBYTES],
                                 uint8_t x, uint8_t y)
{
    uint8_t extseed[KYBER_SYMBYTES + 2];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        extseed[i] = seed[i];
    extseed[KYBER_SYMBYTES] = x;
    extseed[KYBER_SYMBYTES + 1] = y;

    uint64_t state[25];
    shake128_absorb(state, extseed, KYBER_SYMBYTES + 2);

    // Squeeze and reject-sample
    int ctr = 0;
    uint8_t buf[SHAKE128_RATE * 4]; // up to 4 blocks usually enough
    while (ctr < KYBER_N)
    {
        // Squeeze one block
        const uint8_t *s = (const uint8_t *)state;
        for (int i = 0; i < SHAKE128_RATE; i++)
            buf[i] = s[i];
        keccakf1600(state);

        ctr += rej_uniform(entry->coeffs + ctr, KYBER_N - ctr,
                           buf, SHAKE128_RATE);
    }
}

// Generate full k×k matrix Â in NTT domain
// transposed: 0 for Â, 1 for Â^T
__device__ void gen_matrix(polyvec a[KYBER_K], const uint8_t seed[KYBER_SYMBYTES],
                           int transposed)
{
    for (int i = 0; i < KYBER_K; i++)
    {
        for (int j = 0; j < KYBER_K; j++)
        {
            if (transposed)
                gen_matrix_entry(&a[i].vec[j], seed, (uint8_t)i, (uint8_t)j);
            else
                gen_matrix_entry(&a[i].vec[j], seed, (uint8_t)j, (uint8_t)i);
        }
    }
}

// ---- Sample secret/noise vectors via SHAKE-256 + CBD ----
__device__ void sample_poly_cbd_eta1(poly *r, const uint8_t seed[KYBER_SYMBYTES],
                                     uint8_t nonce)
{
    uint8_t extseed[KYBER_SYMBYTES + 1];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        extseed[i] = seed[i];
    extseed[KYBER_SYMBYTES] = nonce;

    uint8_t buf[KYBER_ETA1 * KYBER_N / 4]; // 128 bytes
    uint64_t state[25];
    shake256_absorb(state, extseed, KYBER_SYMBYTES + 1);
    keccak_squeeze(buf, sizeof(buf), state, SHAKE256_RATE);
    cbd_eta1(r, buf);
}

__device__ void sample_poly_cbd_eta2(poly *r, const uint8_t seed[KYBER_SYMBYTES],
                                     uint8_t nonce)
{
    uint8_t extseed[KYBER_SYMBYTES + 1];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        extseed[i] = seed[i];
    extseed[KYBER_SYMBYTES] = nonce;

    uint8_t buf[KYBER_ETA2 * KYBER_N / 4];
    uint64_t state[25];
    shake256_absorb(state, extseed, KYBER_SYMBYTES + 1);
    keccak_squeeze(buf, sizeof(buf), state, SHAKE256_RATE);
    cbd_eta2(r, buf);
}

// ---- Polynomial serialization (Encode/Decode) ----
// poly_tobytes: serialize polynomial to 384 bytes (12 bits per coeff)
__device__ void poly_tobytes(uint8_t r[KYBER_POLYBYTES], const poly *a)
{
    for (int i = 0; i < KYBER_N / 2; i++)
    {
        uint16_t t0 = (uint16_t)a->coeffs[2 * i];
        uint16_t t1 = (uint16_t)a->coeffs[2 * i + 1];
        r[3 * i] = (uint8_t)(t0);
        r[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        r[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
}

// poly_frombytes: deserialize 384 bytes to polynomial
__device__ void poly_frombytes(poly *r, const uint8_t a[KYBER_POLYBYTES])
{
    for (int i = 0; i < KYBER_N / 2; i++)
    {
        r->coeffs[2 * i] = (int16_t)(((uint16_t)a[3 * i] | ((uint16_t)a[3 * i + 1] << 8)) & 0xFFF);
        r->coeffs[2 * i + 1] = (int16_t)(((uint16_t)(a[3 * i + 1] >> 4) | ((uint16_t)a[3 * i + 2] << 4)) & 0xFFF);
    }
}

// polyvec serialization
__device__ void polyvec_tobytes(uint8_t *r, const polyvec *a)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_tobytes(r + i * KYBER_POLYBYTES, &a->vec[i]);
}

__device__ void polyvec_frombytes(polyvec *r, const uint8_t *a)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_frombytes(&r->vec[i], a + i * KYBER_POLYBYTES);
}

// ---- Compression / Decompression ----
// Compress: round(2^d / q * x) mod 2^d
__device__ __forceinline__
    uint16_t
    compress_d(uint16_t x, int d)
{
    uint32_t t = ((uint32_t)x << d) + KYBER_Q / 2;
    // Divide by q using Barrett-like trick
    t = (t * 315) >> 20; // 315/2^20 ≈ 1/3329 * 2^d approximation
    // More precise:
    t = ((uint32_t)x << d) + (KYBER_Q >> 1);
    t = t / KYBER_Q;
    return (uint16_t)(t & ((1 << d) - 1));
}

__device__ __forceinline__
    uint16_t
    decompress_d(uint16_t x, int d)
{
    return (uint16_t)(((uint32_t)x * KYBER_Q + (1 << (d - 1))) >> d);
}

// Compress polynomial (dv=4 for v, du=10 for u)
__device__ void poly_compress_dv(uint8_t r[KYBER_POLYCOMPRESSEDBYTES], const poly *a)
{
    uint8_t t[8];
    for (int i = 0; i < KYBER_N / 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int16_t u = a->coeffs[8 * i + j];
            u += (u >> 15) & KYBER_Q; // make positive
            t[j] = (uint8_t)compress_d((uint16_t)u, KYBER_DV);
        }
        r[4 * i] = t[0] | (t[1] << 4);
        r[4 * i + 1] = t[2] | (t[3] << 4);
        r[4 * i + 2] = t[4] | (t[5] << 4);
        r[4 * i + 3] = t[6] | (t[7] << 4);
    }
}

__device__ void poly_decompress_dv(poly *r, const uint8_t a[KYBER_POLYCOMPRESSEDBYTES])
{
    for (int i = 0; i < KYBER_N / 2; i++)
    {
        r->coeffs[2 * i] = (int16_t)decompress_d(a[i] & 0xF, KYBER_DV);
        r->coeffs[2 * i + 1] = (int16_t)decompress_d(a[i] >> 4, KYBER_DV);
    }
}

// Compress polyvec (du=10)
__device__ void polyvec_compress(uint8_t r[KYBER_POLYVECCOMPRESSEDBYTES], const polyvec *a)
{
    for (int i = 0; i < KYBER_K; i++)
    {
        for (int j = 0; j < KYBER_N / 4; j++)
        {
            uint16_t t[4];
            for (int k = 0; k < 4; k++)
            {
                int16_t u = a->vec[i].coeffs[4 * j + k];
                u += (u >> 15) & KYBER_Q;
                t[k] = compress_d((uint16_t)u, KYBER_DU);
            }
            // Pack 4 × 10-bit values into 5 bytes
            uint8_t *p = r + i * (KYBER_N * KYBER_DU / 8) + 5 * j;
            p[0] = (uint8_t)(t[0]);
            p[1] = (uint8_t)((t[0] >> 8) | (t[1] << 2));
            p[2] = (uint8_t)((t[1] >> 6) | (t[2] << 4));
            p[3] = (uint8_t)((t[2] >> 4) | (t[3] << 6));
            p[4] = (uint8_t)(t[3] >> 2);
        }
    }
}

__device__ void polyvec_decompress(polyvec *r, const uint8_t a[KYBER_POLYVECCOMPRESSEDBYTES])
{
    for (int i = 0; i < KYBER_K; i++)
    {
        for (int j = 0; j < KYBER_N / 4; j++)
        {
            const uint8_t *p = a + i * (KYBER_N * KYBER_DU / 8) + 5 * j;
            uint16_t t[4];
            t[0] = ((uint16_t)p[0] | ((uint16_t)p[1] << 8)) & 0x3FF;
            t[1] = (((uint16_t)p[1] >> 2) | ((uint16_t)p[2] << 6)) & 0x3FF;
            t[2] = (((uint16_t)p[2] >> 4) | ((uint16_t)p[3] << 4)) & 0x3FF;
            t[3] = (((uint16_t)p[3] >> 6) | ((uint16_t)p[4] << 2)) & 0x3FF;
            for (int k = 0; k < 4; k++)
                r->vec[i].coeffs[4 * j + k] = (int16_t)decompress_d(t[k], KYBER_DU);
        }
    }
}

// ---- Message encoding/decoding ----
// Encode 32-byte message as polynomial (each bit -> q/2 or 0)
__device__ void poly_frommsg(poly *r, const uint8_t msg[KYBER_SYMBYTES])
{
    for (int i = 0; i < KYBER_N / 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int16_t mask = -((int16_t)(msg[i] >> j) & 1);
            r->coeffs[8 * i + j] = mask & ((KYBER_Q + 1) / 2);
        }
    }
}

// Decode polynomial to 32-byte message (compress to 1 bit)
__device__ void poly_tomsg(uint8_t msg[KYBER_SYMBYTES], const poly *a)
{
    for (int i = 0; i < KYBER_N / 8; i++)
    {
        msg[i] = 0;
        for (int j = 0; j < 8; j++)
        {
            int16_t u = a->coeffs[8 * i + j];
            u += (u >> 15) & KYBER_Q;
            // t = round(2/q * u) mod 2
            uint16_t t = (((uint16_t)u << 1) + KYBER_Q / 2) / KYBER_Q;
            msg[i] |= (uint8_t)((t & 1) << j);
        }
    }
}

// Pack public key: pk = encode(t_hat) || rho
__device__ void pack_pk(uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES],
                        const polyvec *t_hat, const uint8_t rho[KYBER_SYMBYTES])
{
    polyvec_tobytes(pk, t_hat);
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        pk[KYBER_POLYVECBYTES + i] = rho[i];
}

__device__ void unpack_pk(polyvec *t_hat, uint8_t rho[KYBER_SYMBYTES],
                          const uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES])
{
    polyvec_frombytes(t_hat, pk);
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        rho[i] = pk[KYBER_POLYVECBYTES + i];
}

// Pack secret key
__device__ void pack_sk(uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES], const polyvec *s)
{
    polyvec_tobytes(sk, s);
}

__device__ void unpack_sk(polyvec *s, const uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES])
{
    polyvec_frombytes(s, sk);
}

// Pack ciphertext: c = compress(u) || compress(v)
__device__ void pack_ciphertext(uint8_t ct[KYBER_INDCPA_BYTES],
                                const polyvec *u, const poly *v)
{
    polyvec_compress(ct, u);
    poly_compress_dv(ct + KYBER_POLYVECCOMPRESSEDBYTES, v);
}

__device__ void unpack_ciphertext(polyvec *u, poly *v,
                                  const uint8_t ct[KYBER_INDCPA_BYTES])
{
    polyvec_decompress(u, ct);
    poly_decompress_dv(v, ct + KYBER_POLYVECCOMPRESSEDBYTES);
}

#endif // KYBER_UTILS_CUH
