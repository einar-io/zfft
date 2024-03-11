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

#define N 1024
#define NDEBUG true
#define assertm(exp, msg) assert(((void)msg, exp))

// cuComplex is cuFloatComplex
typedef cuComplex CoefT;
typedef CoefT Poly[N];

__host__ __device__ static __inline__ bool cuCeqf(cuFloatComplex x,
                                                  cuFloatComplex y)
{
    return cuCrealf(x) == cuCrealf(y) && cuCimagf(x) == cuCimagf(y);
}

__global__ void add_kernel(int n, CoefT *dst, const CoefT *src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = cuCaddf(dst[i], src[i]);
}

int main()
{

    Poly p, q, s, d_p, d_q, d_s;

    for (int i = 0; i < N; i++)
    {
        p[i] = make_cuComplex(float(i), float(i));
        q[i] = cuConjf(p[i]);
        s[i] = cuCaddf(p[i], q[i]);
    }

    int n = 1 << 10;
    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);

    // async
    add_kernel<<<blocks, threads>>>(N, d_s, d_p);
    checkCudaErrors(cudaPeekAtLastError());

    for (int i = 0; i < N; i++)
    {
        assertm(cuCeqf(s[i], d_s[i]), "CPU and GPU results differ.");
    }

    std::exit(EXIT_SUCCESS);
}
