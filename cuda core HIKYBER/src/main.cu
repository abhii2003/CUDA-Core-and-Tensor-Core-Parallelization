#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "../include/params.h"
#include "../include/kyber_kem.cuh"

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

static void rand_bytes(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
        buf[i] = (uint8_t)(rand() & 0xFF);
}

static void upload_zetas()
{
    int16_t z[128], zi[128];
    for (int i = 0; i < 128; i++)
    {
        z[i] = h_zetas[i];
        zi[i] = h_zetas_inv[i];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_zetas, z, sizeof(z)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_zetas_inv, zi, sizeof(zi)));
}

// Correctness test: keygen -> encaps -> decaps, verify shared secrets match
static bool test_correctness(int blocks, int tpb)
{
    int batch = blocks * tpb;
    size_t d_sz = (size_t)batch * KYBER_SYMBYTES;
    size_t z_sz = (size_t)batch * KYBER_SYMBYTES;
    size_t pk_sz = (size_t)batch * KYBER_PUBLICKEYBYTES;
    size_t sk_sz = (size_t)batch * KYBER_SECRETKEYBYTES;
    size_t m_sz = (size_t)batch * KYBER_SYMBYTES;
    size_t ct_sz = (size_t)batch * KYBER_CIPHERTEXTBYTES;
    size_t ss_sz = (size_t)batch * KYBER_SSBYTES;

    uint8_t *h_d, *h_z, *h_pk, *h_sk, *h_m, *h_ct, *h_ss_e, *h_ss_d;
    h_d = (uint8_t *)malloc(d_sz);
    h_z = (uint8_t *)malloc(z_sz);
    h_pk = (uint8_t *)malloc(pk_sz);
    h_sk = (uint8_t *)malloc(sk_sz);
    h_m = (uint8_t *)malloc(m_sz);
    h_ct = (uint8_t *)malloc(ct_sz);
    h_ss_e = (uint8_t *)malloc(ss_sz);
    h_ss_d = (uint8_t *)malloc(ss_sz);

    rand_bytes(h_d, d_sz);
    rand_bytes(h_z, z_sz);
    rand_bytes(h_m, m_sz);

    uint8_t *dd, *dz, *dpk, *dsk, *dm, *dct, *dss_e, *dss_d;
    CUDA_CHECK(cudaMalloc(&dd, d_sz));
    CUDA_CHECK(cudaMalloc(&dz, z_sz));
    CUDA_CHECK(cudaMalloc(&dpk, pk_sz));
    CUDA_CHECK(cudaMalloc(&dsk, sk_sz));
    CUDA_CHECK(cudaMalloc(&dm, m_sz));
    CUDA_CHECK(cudaMalloc(&dct, ct_sz));
    CUDA_CHECK(cudaMalloc(&dss_e, ss_sz));
    CUDA_CHECK(cudaMalloc(&dss_d, ss_sz));

    CUDA_CHECK(cudaMemcpy(dd, h_d, d_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, h_z, z_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm, h_m, m_sz, cudaMemcpyHostToDevice));

    kem_keygen_kernel<<<blocks, tpb>>>(dd, dz, dpk, dsk, batch);
    CUDA_CHECK(cudaGetLastError());

    kem_encaps_kernel<<<blocks, tpb>>>(dpk, dm, dct, dss_e, batch);
    CUDA_CHECK(cudaGetLastError());

    kem_decaps_kernel<<<blocks, tpb>>>(dct, dsk, dss_d, batch);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ss_e, dss_e, ss_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ss_d, dss_d, ss_sz, cudaMemcpyDeviceToHost));

    int pass = 0, fail = 0;
    for (int i = 0; i < batch; i++)
    {
        if (memcmp(h_ss_e + i * KYBER_SSBYTES,
                   h_ss_d + i * KYBER_SSBYTES, KYBER_SSBYTES) == 0)
            pass++;
        else
            fail++;
    }

    printf("Correctness: %d/%d passed, %d failed\n", pass, batch, fail);

    cudaFree(dd);
    cudaFree(dz);
    cudaFree(dpk);
    cudaFree(dsk);
    cudaFree(dm);
    cudaFree(dct);
    cudaFree(dss_e);
    cudaFree(dss_d);
    free(h_d);
    free(h_z);
    free(h_pk);
    free(h_sk);
    free(h_m);
    free(h_ct);
    free(h_ss_e);
    free(h_ss_d);

    return fail == 0;
}

// Benchmark a single kernel
struct BenchResult
{
    float ms;
    double ops_per_sec;
};

static BenchResult bench_keygen(int blocks, int tpb, int iters)
{
    int batch = blocks * tpb;
    uint8_t *dd, *dz, *dpk, *dsk;
    CUDA_CHECK(cudaMalloc(&dd, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dz, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dpk, (size_t)batch * KYBER_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&dsk, (size_t)batch * KYBER_SECRETKEYBYTES));

    uint8_t *h_buf = (uint8_t *)malloc((size_t)batch * KYBER_SYMBYTES);
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dd, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dz, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    free(h_buf);

    // Warmup
    kem_keygen_kernel<<<blocks, tpb>>>(dd, dz, dpk, dsk, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kem_keygen_kernel<<<blocks, tpb>>>(dd, dz, dpk, dsk, batch);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dd);
    cudaFree(dz);
    cudaFree(dpk);
    cudaFree(dsk);

    double total_ops = (double)batch * iters;
    double ops = total_ops / (ms / 1000.0);
    return {ms / iters, ops};
}

static BenchResult bench_encaps(int blocks, int tpb, int iters)
{
    int batch = blocks * tpb;
    uint8_t *dd, *dz, *dpk, *dsk, *dm, *dct, *dss;
    CUDA_CHECK(cudaMalloc(&dd, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dz, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dpk, (size_t)batch * KYBER_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&dsk, (size_t)batch * KYBER_SECRETKEYBYTES));
    CUDA_CHECK(cudaMalloc(&dm, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dct, (size_t)batch * KYBER_CIPHERTEXTBYTES));
    CUDA_CHECK(cudaMalloc(&dss, (size_t)batch * KYBER_SSBYTES));

    uint8_t *h_buf = (uint8_t *)malloc((size_t)batch * KYBER_SYMBYTES);
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dd, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dz, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    free(h_buf);

    kem_keygen_kernel<<<blocks, tpb>>>(dd, dz, dpk, dsk, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    h_buf = (uint8_t *)malloc((size_t)batch * KYBER_SYMBYTES);
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dm, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    free(h_buf);

    // Warmup
    kem_encaps_kernel<<<blocks, tpb>>>(dpk, dm, dct, dss, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kem_encaps_kernel<<<blocks, tpb>>>(dpk, dm, dct, dss, batch);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dd);
    cudaFree(dz);
    cudaFree(dpk);
    cudaFree(dsk);
    cudaFree(dm);
    cudaFree(dct);
    cudaFree(dss);

    double total_ops = (double)batch * iters;
    double ops = total_ops / (ms / 1000.0);
    return {ms / iters, ops};
}

static BenchResult bench_decaps(int blocks, int tpb, int iters)
{
    int batch = blocks * tpb;
    uint8_t *dd, *dz, *dpk, *dsk, *dm, *dct, *dss_e, *dss_d;
    CUDA_CHECK(cudaMalloc(&dd, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dz, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dpk, (size_t)batch * KYBER_PUBLICKEYBYTES));
    CUDA_CHECK(cudaMalloc(&dsk, (size_t)batch * KYBER_SECRETKEYBYTES));
    CUDA_CHECK(cudaMalloc(&dm, (size_t)batch * KYBER_SYMBYTES));
    CUDA_CHECK(cudaMalloc(&dct, (size_t)batch * KYBER_CIPHERTEXTBYTES));
    CUDA_CHECK(cudaMalloc(&dss_e, (size_t)batch * KYBER_SSBYTES));
    CUDA_CHECK(cudaMalloc(&dss_d, (size_t)batch * KYBER_SSBYTES));

    uint8_t *h_buf = (uint8_t *)malloc((size_t)batch * KYBER_SYMBYTES);
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dd, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dz, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    free(h_buf);

    kem_keygen_kernel<<<blocks, tpb>>>(dd, dz, dpk, dsk, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    h_buf = (uint8_t *)malloc((size_t)batch * KYBER_SYMBYTES);
    rand_bytes(h_buf, (size_t)batch * KYBER_SYMBYTES);
    CUDA_CHECK(cudaMemcpy(dm, h_buf, (size_t)batch * KYBER_SYMBYTES, cudaMemcpyHostToDevice));
    free(h_buf);

    kem_encaps_kernel<<<blocks, tpb>>>(dpk, dm, dct, dss_e, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    kem_decaps_kernel<<<blocks, tpb>>>(dct, dsk, dss_d, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kem_decaps_kernel<<<blocks, tpb>>>(dct, dsk, dss_d, batch);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dd);
    cudaFree(dz);
    cudaFree(dpk);
    cudaFree(dsk);
    cudaFree(dm);
    cudaFree(dct);
    cudaFree(dss_e);
    cudaFree(dss_d);

    double total_ops = (double)batch * iters;
    double ops = total_ops / (ms / 1000.0);
    return {ms / iters, ops};
}

int main(int argc, char **argv)
{
    srand(42);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  SMs: %d  MaxThreads/SM: %d  CC: %d.%d\n",
           prop.name, prop.multiProcessorCount,
           prop.maxThreadsPerMultiProcessor,
           prop.major, prop.minor);

    upload_zetas();

    int blocks = DEFAULT_BLOCKS;
    int tpb = DEFAULT_THREADS_PER_BLOCK;
    int iters = 10;
    bool do_sweep = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
            blocks = atoi(argv[++i]);
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            tpb = atoi(argv[++i]);
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iters = atoi(argv[++i]);
        if (strcmp(argv[i], "--sweep") == 0)
            do_sweep = true;
    }

    printf("\n=== Correctness Test (batch=%d) ===\n", blocks * tpb);
    bool ok = test_correctness(blocks, tpb);
    if (!ok)
    {
        printf("FAIL: shared secrets mismatch, aborting benchmark.\n");
        return 1;
    }

    if (do_sweep)
    {
        int blk_opts[] = {10, 20, 32, 40};
        int tpb_opts[] = {64, 128, 256};
        printf("\n=== Parameter Sweep ===\n");
        printf("%6s %6s %8s %12s %12s %12s %12s %12s\n",
               "Blocks", "TPB", "Batch",
               "KG ops/s", "Enc ops/s", "Dec ops/s",
               "KX ops/s", "Thrpt ops/s");
        for (int bi = 0; bi < 4; bi++)
        {
            for (int ti = 0; ti < 3; ti++)
            {
                int b = blk_opts[bi], t = tpb_opts[ti];
                int n = b * t;
                auto kg = bench_keygen(b, t, iters);
                auto en = bench_encaps(b, t, iters);
                auto de = bench_decaps(b, t, iters);
                double kx = (kg.ops_per_sec * de.ops_per_sec) /
                            (kg.ops_per_sec + de.ops_per_sec);
                double thrpt = 1.0 / (1.0 / kg.ops_per_sec + 1.0 / en.ops_per_sec + 1.0 / de.ops_per_sec);
                printf("%6d %6d %8d %12.0f %12.0f %12.0f %12.0f %12.0f\n",
                       b, t, n, kg.ops_per_sec, en.ops_per_sec, de.ops_per_sec, kx, thrpt);
            }
        }
    }
    else
    {
        printf("\n=== Benchmark (blocks=%d, tpb=%d, iters=%d) ===\n", blocks, tpb, iters);
        auto kg = bench_keygen(blocks, tpb, iters);
        auto en = bench_encaps(blocks, tpb, iters);
        auto de = bench_decaps(blocks, tpb, iters);
        double kx = (kg.ops_per_sec * de.ops_per_sec) /
                    (kg.ops_per_sec + de.ops_per_sec);
        printf("KeyGen:  %.3f ms/batch  %.0f ops/s\n", kg.ms, kg.ops_per_sec);
        printf("Encaps:  %.3f ms/batch  %.0f ops/s\n", en.ms, en.ops_per_sec);
        printf("Decaps:  %.3f ms/batch  %.0f ops/s\n", de.ms, de.ops_per_sec);
        printf("KX (KeyGen*Decaps)/(KeyGen+Decaps): %.0f ops/s\n", kx);
    }

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
