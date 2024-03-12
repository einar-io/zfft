#include <cuComplex.h>
#include <cstdlib>
#include <cassert>
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

#define CEIL_DIV(x, y) (((x) + ((y)-1)) / (y))

#define N 1024
#define DEBUG true
#define assertm(exp, msg) assert(((void)msg, exp))

// cuComplex is cuFloatComplex
typedef cuComplex CoefT;
// typedef CoefT Poly[N];

__host__ __device__ static __inline__ bool cuCeqf(cuFloatComplex x,
                                                  cuFloatComplex y)
{
    // todo: better float comparison
    return cuCrealf(x) == cuCrealf(y) && cuCimagf(x) == cuCimagf(y);
}

__host__ bool verify(const unsigned int n, const CoefT *h_s, const CoefT *d_s)
{
    for (int i = 0; i < n; i++)
    {
        bool compare = cuCeqf(h_s[i], d_s[i]);
        assertm(compare, "CPU and GPU results differ.");
        if (!compare)
            return false;
    }
    return true;
}

__global__ void add_kernel(int n, CoefT *dst, const CoefT *src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = cuCaddf(dst[i], src[i]);
}

int main()
{
    // output device info and transfer size
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s (device memory: %.1f GiB)\n", prop.name, prop.totalGlobalMem / double(1 << 30));

    // Phase 0: Sizing
    const unsigned int nElements = N;
    const size_t nBytes = nElements * sizeof(cuComplex);
    // printf("poly: %i", sizeof(Poly));
    assertm(sizeof(CoefT) == 2 * sizeof(float), "size of complex");

    printf("Transfer size (MB): %d\n", nBytes / (1024 * 1024));

    // Phase 1: Allocation

    // Create non-default stream for Async
    cudaStream_t stream1;
    checkCudaErrors(cudaStreamCreate(&stream1));

    CoefT *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

    checkCudaErrors(cudaMallocHost((void **)&h_p, nBytes));
    checkCudaErrors(cudaMallocHost((void **)&h_q, nBytes));
    checkCudaErrors(cudaMallocHost((void **)&h_s, nBytes));
    checkCudaErrors(cudaMallocHost((void **)&dh_s, nBytes));

    checkCudaErrors(cudaMalloc((void **)&d_p, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_q, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_s, nBytes));

    // memset
    checkCudaErrors(cudaMemset((void *)d_p, 0, nBytes));
    checkCudaErrors(cudaMemset((void *)d_q, 0, nBytes));
    checkCudaErrors(cudaMemset((void *)d_s, 0, nBytes));

    // Values
    for (int i = 0; i < nElements; i++)
    {
        h_p[i] = make_cuComplex(float(i), float(i));
        h_q[i] = cuConjf(h_p[i]);
        h_s[i] = cuCaddf(h_p[i], h_q[i]);
        // printf("s[%i]: %f + %fi\n", i, h_s[i].x, h_s[i].y);
    }

    // Phase 2: Memcpy: Host-to-Device
    checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));
    checkCudaErrors(cudaMemcpyAsync(d_q, h_q, nBytes, cudaMemcpyHostToDevice, stream1));

    // Phase 3: Execute
    // set kernel launch configuration
    dim3 blockdim = dim3(32, 1);
    dim3 griddim = dim3(CEIL_DIV(nElements, blockdim.x), 1);
    add_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, (CoefT *)d_s, (CoefT *)d_p);
    checkCudaErrors(cudaPeekAtLastError());
    add_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, (CoefT *)d_s, (CoefT *)d_q);
    checkCudaErrors(cudaPeekAtLastError());

    // Phase 4: Memcpy: Device-to-Host
    checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

    // Phase 6:  Verification
    checkCudaErrors(cudaStreamSynchronize(stream1));
    printf("Verify\n");
    bool verified = verify(nElements, dh_s, h_s);
    printf("Verification %s.\n", verified ? "succeeded" : "failed");

    // Phase 7: Deallocation
    checkCudaErrors(cudaFreeHost(h_p));
    checkCudaErrors(cudaFreeHost(h_q));
    checkCudaErrors(cudaFreeHost(h_s));
    checkCudaErrors(cudaFreeHost(dh_s));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_q));
    checkCudaErrors(cudaFree(d_s));
    checkCudaErrors(cudaStreamDestroy(stream1));

    std::exit(verified ? EXIT_SUCCESS : EXIT_FAILURE);
}
