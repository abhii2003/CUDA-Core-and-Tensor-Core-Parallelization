#ifndef KYBER_KEM_CUH
#define KYBER_KEM_CUH

#include "params.h"
#include "kyber_pke.cuh"

// Fujisaki-Okamoto CCA2-secure KEM wrapper
// KEM secret key layout: sk_cpa || pk || H(pk) || z
// Total: KYBER_INDCPA_SECRETKEYBYTES + KYBER_INDCPA_PUBLICKEYBYTES + 2*KYBER_SYMBYTES

// ---------- KEM KeyGen ----------
// Input:  d[32], z[32] random bytes
// Output: pk[KYBER_PUBLICKEYBYTES], sk_kem[KYBER_SECRETKEYBYTES]
__global__ void kem_keygen_kernel(const uint8_t *d_in,
                                  const uint8_t *z_in,
                                  uint8_t *pk_out,
                                  uint8_t *sk_kem_out,
                                  int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *d = d_in + tid * KYBER_SYMBYTES;
    const uint8_t *z = z_in + tid * KYBER_SYMBYTES;
    uint8_t *pk = pk_out + tid * KYBER_PUBLICKEYBYTES;
    uint8_t *sk = sk_kem_out + tid * KYBER_SECRETKEYBYTES;

    // CPA KeyGen inline (fused)
    uint8_t buf[64];
    sha3_512(buf, d, KYBER_SYMBYTES);
    uint8_t rho[KYBER_SYMBYTES], sigma[KYBER_SYMBYTES];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
    {
        rho[i] = buf[i];
        sigma[i] = buf[KYBER_SYMBYTES + i];
    }

    polyvec a_hat[KYBER_K];
    gen_matrix(a_hat, rho, 0);

    polyvec s, e;
    uint8_t nonce = 0;
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&s.vec[i], sigma, nonce++);
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&e.vec[i], sigma, nonce++);

    polyvec_ntt(&s);
    polyvec_ntt(&e);

    polyvec t_hat;
    for (int i = 0; i < KYBER_K; i++)
    {
        polyvec_basemul_acc(&t_hat.vec[i], &a_hat[i], &s);
        poly_add(&t_hat.vec[i], &t_hat.vec[i], &e.vec[i]);
    }
    polyvec_reduce(&t_hat);

    // Pack pk
    pack_pk(pk, &t_hat, rho);

    // Pack KEM sk: sk_cpa || pk || H(pk) || z
    pack_sk(sk, &s);

    // Copy pk into sk
    for (int i = 0; i < KYBER_PUBLICKEYBYTES; i++)
        sk[KYBER_INDCPA_SECRETKEYBYTES + i] = pk[i];

    // H(pk) into sk
    sha3_256(sk + KYBER_INDCPA_SECRETKEYBYTES + KYBER_PUBLICKEYBYTES,
             pk, KYBER_PUBLICKEYBYTES);

    // z into sk
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        sk[KYBER_SECRETKEYBYTES - KYBER_SYMBYTES + i] = z[i];
}

// ---------- KEM Encaps ----------
// Input:  pk, m[32] random message
// Output: ct[KYBER_CIPHERTEXTBYTES], ss[KYBER_SSBYTES] shared secret
__global__ void kem_encaps_kernel(const uint8_t *pk_in,
                                  const uint8_t *m_in,
                                  uint8_t *ct_out,
                                  uint8_t *ss_out,
                                  int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *pk = pk_in + tid * KYBER_PUBLICKEYBYTES;
    const uint8_t *m = m_in + tid * KYBER_SYMBYTES;
    uint8_t *ct = ct_out + tid * KYBER_CIPHERTEXTBYTES;
    uint8_t *ss = ss_out + tid * KYBER_SSBYTES;

    // H(pk)
    uint8_t h_pk[32];
    sha3_256(h_pk, pk, KYBER_PUBLICKEYBYTES);

    // (K_bar, r) = G(m || H(pk))
    uint8_t kr_input[64]; // m || H(pk)
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kr_input[i] = m[i];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kr_input[KYBER_SYMBYTES + i] = h_pk[i];

    uint8_t kr[64]; // K_bar || r
    sha3_512(kr, kr_input, 64);

    // CPA Encrypt inline (fused) with coins = r (kr+32)
    polyvec t_hat;
    uint8_t rho[KYBER_SYMBYTES];
    unpack_pk(&t_hat, rho, pk);

    polyvec a_hat[KYBER_K];
    gen_matrix(a_hat, rho, 1);

    polyvec rv, e1;
    poly e2;
    uint8_t enc_nonce = 0;
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&rv.vec[i], kr + KYBER_SYMBYTES, enc_nonce++);
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta2(&e1.vec[i], kr + KYBER_SYMBYTES, enc_nonce++);
    sample_poly_cbd_eta2(&e2, kr + KYBER_SYMBYTES, enc_nonce++);

    polyvec_ntt(&rv);

    polyvec u;
    for (int i = 0; i < KYBER_K; i++)
        polyvec_basemul_acc(&u.vec[i], &a_hat[i], &rv);
    polyvec_intt(&u);
    polyvec_add(&u, &u, &e1);
    polyvec_reduce(&u);

    poly v;
    polyvec_basemul_acc(&v, &t_hat, &rv);
    poly_intt(&v);
    poly_add(&v, &v, &e2);
    poly mp;
    poly_frommsg(&mp, m);
    poly_add(&v, &v, &mp);
    poly_reduce(&v);

    polyvec_csubq(&u);
    poly_csubq(&v);
    pack_ciphertext(ct, &u, &v);

    // K = KDF(K_bar || H(c))
    uint8_t h_ct[32];
    sha3_256(h_ct, ct, KYBER_CIPHERTEXTBYTES);
    uint8_t kdf_input[64];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kdf_input[i] = kr[i]; // K_bar
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kdf_input[KYBER_SYMBYTES + i] = h_ct[i];
    shake256(ss, KYBER_SSBYTES, kdf_input, 64);
}

// ---------- KEM Decaps ----------
// Input:  ct, sk_kem
// Output: ss[KYBER_SSBYTES] shared secret
__global__ void kem_decaps_kernel(const uint8_t *ct_in,
                                  const uint8_t *sk_kem_in,
                                  uint8_t *ss_out,
                                  int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    const uint8_t *ct = ct_in + tid * KYBER_CIPHERTEXTBYTES;
    const uint8_t *sk = sk_kem_in + tid * KYBER_SECRETKEYBYTES;
    uint8_t *ss = ss_out + tid * KYBER_SSBYTES;

    // Extract components from KEM sk
    const uint8_t *sk_cpa = sk;
    const uint8_t *pk = sk + KYBER_INDCPA_SECRETKEYBYTES;
    const uint8_t *h_pk = pk + KYBER_PUBLICKEYBYTES;
    const uint8_t *z = h_pk + KYBER_SYMBYTES;

    // CPA Decrypt (inline fused)
    polyvec u_dec;
    poly v_dec;
    unpack_ciphertext(&u_dec, &v_dec, ct);

    polyvec s_hat;
    unpack_sk(&s_hat, sk_cpa);

    polyvec_ntt(&u_dec);
    poly tmp;
    polyvec_basemul_acc(&tmp, &s_hat, &u_dec);
    poly_intt(&tmp);
    poly_sub(&v_dec, &v_dec, &tmp);
    poly_reduce(&v_dec);
    poly_csubq(&v_dec);

    uint8_t m_dec[KYBER_SYMBYTES];
    poly_tomsg(m_dec, &v_dec);

    // (K_bar', r') = G(m' || H(pk))
    uint8_t kr_input[64];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kr_input[i] = m_dec[i];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kr_input[KYBER_SYMBYTES + i] = h_pk[i];

    uint8_t kr[64];
    sha3_512(kr, kr_input, 64);

    // Re-encrypt: CPA Encrypt with m' and r'
    polyvec t_hat;
    uint8_t rho[KYBER_SYMBYTES];
    unpack_pk(&t_hat, rho, pk);

    polyvec a_hat[KYBER_K];
    gen_matrix(a_hat, rho, 1);

    polyvec rv, e1;
    poly e2;
    uint8_t enc_nonce = 0;
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta1(&rv.vec[i], kr + KYBER_SYMBYTES, enc_nonce++);
    for (int i = 0; i < KYBER_K; i++)
        sample_poly_cbd_eta2(&e1.vec[i], kr + KYBER_SYMBYTES, enc_nonce++);
    sample_poly_cbd_eta2(&e2, kr + KYBER_SYMBYTES, enc_nonce++);

    polyvec_ntt(&rv);

    polyvec u_enc;
    for (int i = 0; i < KYBER_K; i++)
        polyvec_basemul_acc(&u_enc.vec[i], &a_hat[i], &rv);
    polyvec_intt(&u_enc);
    polyvec_add(&u_enc, &u_enc, &e1);
    polyvec_reduce(&u_enc);

    poly v_enc;
    polyvec_basemul_acc(&v_enc, &t_hat, &rv);
    poly_intt(&v_enc);
    poly_add(&v_enc, &v_enc, &e2);
    poly mp;
    poly_frommsg(&mp, m_dec);
    poly_add(&v_enc, &v_enc, &mp);
    poly_reduce(&v_enc);

    polyvec_csubq(&u_enc);
    poly_csubq(&v_enc);

    uint8_t ct_cmp[KYBER_CIPHERTEXTBYTES];
    pack_ciphertext(ct_cmp, &u_enc, &v_enc);

    // Compare ct with re-encrypted ct' (constant-time)
    uint8_t fail = 0;
    for (int i = 0; i < KYBER_CIPHERTEXTBYTES; i++)
        fail |= ct[i] ^ ct_cmp[i];
    fail = (-(uint16_t)(fail != 0)) >> 8; // 0xFF if fail, 0x00 if match

    // H(c)
    uint8_t h_ct[32];
    sha3_256(h_ct, ct, KYBER_CIPHERTEXTBYTES);

    // K = KDF(K_bar || H(c)) if match, else K = KDF(z || H(c))
    uint8_t kdf_input[64];
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kdf_input[i] = (uint8_t)((~fail & kr[i]) | (fail & z[i]));
    for (int i = 0; i < KYBER_SYMBYTES; i++)
        kdf_input[KYBER_SYMBYTES + i] = h_ct[i];

    shake256(ss, KYBER_SSBYTES, kdf_input, 64);
}

#endif // KYBER_KEM_CUH
