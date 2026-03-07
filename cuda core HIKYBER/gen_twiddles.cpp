#include <cstdio>
#include <cstdint>
#include <cmath>

static const int Q = 3329;
static const int N = 256;
static const int ZETA = 17; // primitive 512th root of unity mod q

int power_mod(int base, int exp, int mod) {
    long long result = 1;
    long long b = ((base % mod) + mod) % mod;
    int e = exp;
    if (e < 0) { // modular inverse via Fermat
        b = power_mod((int)b, mod - 2, mod);
        e = -e;
    }
    for (; e > 0; e >>= 1) {
        if (e & 1) result = result * b % mod;
        b = b * b % mod;
    }
    return (int)result;
}

int br7(int i) {
    int r = 0;
    for (int b = 0; b < 7; b++)
        r |= ((i >> b) & 1) << (6 - b);
    return r;
}

int center(int v) {
    v %= Q;
    if (v < 0) v += Q;
    if (v > Q / 2) v -= Q;
    return v;
}

int main() {
    int16_t zetas[128];
    for (int i = 0; i < 128; i++)
        zetas[i] = (int16_t)center(power_mod(ZETA, br7(i), Q));

    printf("static const int16_t h_zetas[128] = {\n");
    for (int i = 0; i < 128; i++) {
        if (i % 8 == 0) printf("    ");
        printf("%6d,", zetas[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("};\n\n");

    // Inverse zetas for INTT
    int16_t izetas[128];
    for (int i = 0; i < 128; i++)
        izetas[i] = (int16_t)center(-power_mod(ZETA, -(br7(127 - i) + 1), Q));

    // Wait - let's match Kyber reference exactly:
    // INTT uses zetas in reverse: izetas[i] = -zetas[127-i]
    for (int i = 0; i < 128; i++)
        izetas[i] = (int16_t)center(-zetas[127 - i]);

    printf("static const int16_t h_zetas_inv[128] = {\n");
    for (int i = 0; i < 128; i++) {
        if (i % 8 == 0) printf("    ");
        printf("%6d,", izetas[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("};\n\n");

    // Montgomery domain: zetas[i] * 2^16 mod q
    int R = 1 << 16;
    int16_t zm[128];
    for (int i = 0; i < 128; i++)
        zm[i] = (int16_t)center((int)((long long)power_mod(ZETA, br7(i), Q) * R % Q));

    printf("static const int16_t h_zetas_mont[128] = {\n");
    for (int i = 0; i < 128; i++) {
        if (i % 8 == 0) printf("    ");
        printf("%6d,", zm[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("};\n\n");

    // Inverse Montgomery zetas
    int16_t izm[128];
    for (int i = 0; i < 128; i++)
        izm[i] = (int16_t)center((int)(-(long long)power_mod(ZETA, br7(127 - i), Q) * R % Q));

    // Actually just negate the mont zetas in reverse
    for (int i = 0; i < 128; i++)
        izm[i] = (int16_t)center(-zm[127 - i]);

    printf("static const int16_t h_zetas_inv_mont[128] = {\n");
    for (int i = 0; i < 128; i++) {
        if (i % 8 == 0) printf("    ");
        printf("%6d,", izm[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("};\n\n");

    // n^{-1} mod q
    int ninv = power_mod(256, Q - 2, Q);
    int ninv_mont = center((int)((long long)ninv * R % Q));
    printf("// n^{-1} mod q = %d\n", ninv);
    printf("// n^{-1} * 2^16 mod q = %d\n", ninv_mont);
    return 0;
}
