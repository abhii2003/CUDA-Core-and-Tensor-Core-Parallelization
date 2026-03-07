#ifndef KECCAK_CUH
#define KECCAK_CUH

#include <cstdint>

// Keccak-f[1600] round constants
__device__ static const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808aULL,
    0x8000000080008000ULL,
    0x000000000000808bULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008aULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000aULL,
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL,
};

// Rotation offsets
__device__ static const int keccak_rot[25] = {
    0,
    1,
    62,
    28,
    27,
    36,
    44,
    6,
    55,
    20,
    3,
    10,
    43,
    25,
    39,
    41,
    45,
    15,
    21,
    8,
    18,
    2,
    61,
    56,
    14,
};

// Pi permutation indices
__device__ static const int keccak_pi[25] = {
    0,
    10,
    7,
    11,
    17,
    20,
    4,
    3,
    5,
    16,
    1,
    8,
    13,
    12,
    14,
    15,
    9,
    2,
    6,
    23,
    18,
    19,
    22,
    24,
    21,
};

__device__ __forceinline__
    uint64_t
    rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

__device__ void keccakf1600(uint64_t state[25])
{
    uint64_t t, bc[5];

    for (int round = 0; round < 24; round++)
    {
        // Theta
        for (int i = 0; i < 5; i++)
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        for (int i = 0; i < 5; i++)
        {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                state[j + i] ^= t;
        }

        // Rho + Pi
        uint64_t tmp[25];
        for (int i = 0; i < 25; i++)
            tmp[keccak_pi[i]] = rotl64(state[i], keccak_rot[i]);

        // Chi
        for (int j = 0; j < 25; j += 5)
        {
            for (int i = 0; i < 5; i++)
                state[j + i] = tmp[j + i] ^ (~tmp[j + (i + 1) % 5] & tmp[j + (i + 2) % 5]);
        }

        // Iota
        state[0] ^= keccak_rc[round];
    }
}

// --- Absorb / squeeze helpers ---

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72

__device__ void keccak_init(uint64_t state[25])
{
    for (int i = 0; i < 25; i++)
        state[i] = 0;
}

__device__ void keccak_absorb(uint64_t state[25], int rate,
                              const uint8_t *in, int inlen, uint8_t dsep)
{
    while (inlen >= rate)
    {
        for (int i = 0; i < rate / 8; i++)
            state[i] ^= ((const uint64_t *)in)[i];
        keccakf1600(state);
        in += rate;
        inlen -= rate;
    }
    // Partial block + padding
    uint8_t buf[200] = {0};
    for (int i = 0; i < inlen; i++)
        buf[i] = in[i];
    buf[inlen] = dsep;
    buf[rate - 1] |= 0x80;
    for (int i = 0; i < rate / 8; i++)
        state[i] ^= ((uint64_t *)buf)[i];
    keccakf1600(state);
}

__device__ void keccak_squeeze(uint8_t *out, int outlen,
                               uint64_t state[25], int rate)
{
    while (outlen > 0)
    {
        int chunk = outlen < rate ? outlen : rate;
        const uint8_t *s = (const uint8_t *)state;
        for (int i = 0; i < chunk; i++)
            out[i] = s[i];
        out += chunk;
        outlen -= chunk;
        if (outlen > 0)
            keccakf1600(state);
    }
}

// --- High-level SHA3 / SHAKE ---

__device__ void sha3_256(uint8_t out[32], const uint8_t *in, int inlen)
{
    uint64_t state[25];
    keccak_init(state);
    keccak_absorb(state, SHA3_256_RATE, in, inlen, 0x06);
    keccak_squeeze(out, 32, state, SHA3_256_RATE);
}

__device__ void sha3_512(uint8_t out[64], const uint8_t *in, int inlen)
{
    uint64_t state[25];
    keccak_init(state);
    keccak_absorb(state, SHA3_512_RATE, in, inlen, 0x06);
    keccak_squeeze(out, 64, state, SHA3_512_RATE);
}

// SHAKE-128: absorb then squeeze incrementally
__device__ void shake128_absorb(uint64_t state[25], const uint8_t *in, int inlen)
{
    keccak_init(state);
    keccak_absorb(state, SHAKE128_RATE, in, inlen, 0x1F);
}

__device__ void shake128_squeeze(uint8_t *out, int nblocks, uint64_t state[25])
{
    for (int i = 0; i < nblocks; i++)
    {
        const uint8_t *s = (const uint8_t *)state;
        for (int j = 0; j < SHAKE128_RATE; j++)
            out[j] = s[j];
        out += SHAKE128_RATE;
        keccakf1600(state);
    }
}

// SHAKE-256
__device__ void shake256_absorb(uint64_t state[25], const uint8_t *in, int inlen)
{
    keccak_init(state);
    keccak_absorb(state, SHAKE256_RATE, in, inlen, 0x1F);
}

__device__ void shake256_squeeze(uint8_t *out, int nblocks, uint64_t state[25])
{
    for (int i = 0; i < nblocks; i++)
    {
        const uint8_t *s = (const uint8_t *)state;
        for (int j = 0; j < SHAKE256_RATE; j++)
            out[j] = s[j];
        out += SHAKE256_RATE;
        keccakf1600(state);
    }
}

// SHAKE-256 one-shot (absorb + squeeze all at once)
__device__ void shake256(uint8_t *out, int outlen, const uint8_t *in, int inlen)
{
    uint64_t state[25];
    shake256_absorb(state, in, inlen);
    keccak_squeeze(out, outlen, state, SHAKE256_RATE);
}

#endif // KECCAK_CUH
