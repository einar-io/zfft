#include <helper_cuda.h>
#include <vector>
#include "../utils.cuh"
#include <cuComplex.h>
#include "gpu.cuh"
#include "driver_types.h"
#include "../include/helper_cuda.h"

namespace gpu
{
    using namespace std;
    /*
    void mul_pinned()
    {

        // Phase 1: Allocation

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        cuComplex *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

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
        // (1x^2 + 1x + 1) * (1x^2 + 1x + 1) = 1x^4 + 2x^3 + 3x^2 + 2x + 1
        // [1,1,1] * [1,1,1] = [1,2,3,2,1]
        h_p[0] = make_cuComplex(float(1), float(0));
        h_p[1] = make_cuComplex(float(1), float(0));
        h_p[2] = make_cuComplex(float(1), float(0));
        h_p[3] = make_cuComplex(float(0), float(0));
        h_p[4] = make_cuComplex(float(0), float(0));
        h_p[5] = make_cuComplex(float(0), float(0));
        h_p[6] = make_cuComplex(float(0), float(0));
        h_p[7] = make_cuComplex(float(0), float(0));

        h_q[0] = make_cuComplex(float(1), float(0));
        h_q[1] = make_cuComplex(float(1), float(0));
        h_q[2] = make_cuComplex(float(1), float(0));
        h_q[3] = make_cuComplex(float(0), float(0));
        h_q[4] = make_cuComplex(float(0), float(0));
        h_q[5] = make_cuComplex(float(0), float(0));
        h_q[6] = make_cuComplex(float(0), float(0));
        h_q[7] = make_cuComplex(float(0), float(0));

        // Phase 2: Memcpy: Host-to-Device
        checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));
        checkCudaErrors(cudaMemcpyAsync(d_q, h_q, nBytes, cudaMemcpyHostToDevice, stream1));

        // Phase 3: Execute
        // set kernel launch configuration

        // start
        dim3 blockdim = dim3(32, 1);
        dim3 griddim = dim3(CEIL_DIV(nElements, blockdim.x), 1);
        add_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());

        mul_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, log_n, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());
        // end

    dim3 blockdim = dim3(32, 1);
    dim3 griddim = dim3(CEIL_DIV(32, blockdim.x), 1);
    // fftGPU<<<griddim, blockdim, 0, stream1>>>(4, 2, (cuComplex *)d_p);
    // mul_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, log_n, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
    // pad
    gpu::fft<<<griddim, blockdim, 0, stream1>>>(8, 3, (cuComplex *)d_p);
    checkCudaErrors(cudaPeekAtLastError());
    gpu::fft<<<griddim, blockdim, 0, stream1>>>(8, 3, (cuComplex *)d_q);
    checkCudaErrors(cudaPeekAtLastError());
    // sync
    gpu::pvMul<<<griddim, blockdim, 0, stream1>>>(8, d_s, d_p, d_q);
    checkCudaErrors(cudaPeekAtLastError());
    // ifft
    gpu::fft<<<griddim, blockdim, 0, stream1>>>(8, 3, (cuComplex *)d_s, true);
    checkCudaErrors(cudaPeekAtLastError());

    // Phase 4: Memcpy: Device-to-Host
    checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

    // Phase 6:  Verification
    printf("Verify\n");
    checkCudaErrors(cudaStreamSynchronize(stream1));
    for (int i = 0; i < 8; i++)
    {
        printf("s[%i]: %f + %fi\n", i, dh_s[i].x, dh_s[i].y);
    }
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
    checkCudaErrors(cudaDeviceReset());
}
*/

    vector<cuComplex> mul_simple(const vector<cuComplex> &p_orig, const vector<cuComplex> &q_orig)
    {
        vector<cuComplex> p(p_orig.begin(), p_orig.end());
        vector<cuComplex> q(q_orig.begin(), q_orig.end());

        ASSERT_MSG(is_power_of_two(p.size()), "Must be a power of two.");
        ASSERT_MSG(p.size() == q.size(), "Must have same size.");

        int old = p.size();
        // pad
        int n = p.size() + q.size();
        int log_n = log_2(n);
        ASSERT_MSG((1 << log_n) == n, "2^log_n = n.");

        p.resize(n);
        q.resize(n);

        cuComplex zero = make_cuComplex(0.0f, 0.0f);
        for (int i = old; i < p.size(); i++)
        {
            p[i] = zero;
            q[i] = zero;
        }

        vector<cuComplex> s;
        s.resize(n);

        // Phase 0: Sizing
        int nElements = n;
        int nBytes = n * sizeof(cuComplex);

        // Phase 1: Allocation

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        cuComplex *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

        h_p = p.data();
        h_q = q.data();
        dh_s = s.data();

        checkCudaErrors(cudaMalloc((void **)&d_p, nBytes));
        checkCudaErrors(cudaMalloc((void **)&d_q, nBytes));
        checkCudaErrors(cudaMalloc((void **)&d_s, nBytes));

        // memset
        checkCudaErrors(cudaMemset((void *)d_p, 0, nBytes));
        checkCudaErrors(cudaMemset((void *)d_q, 0, nBytes));
        checkCudaErrors(cudaMemset((void *)d_s, 0, nBytes));

        // Phase 2: Memcpy: Host-to-Device
        checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));
        checkCudaErrors(cudaMemcpyAsync(d_q, h_q, nBytes, cudaMemcpyHostToDevice, stream1));

        // Phase 3: Execute
        // set kernel launch configuration

        dim3 blockdim = dim3(32, 1);
        dim3 griddim = dim3(CEIL_DIV(32, blockdim.x), 1);

        gpu::fft<<<griddim, blockdim, 0, stream1>>>(n, log_n, (cuComplex *)d_p);
        checkCudaErrors(cudaPeekAtLastError());
        gpu::fft<<<griddim, blockdim, 0, stream1>>>(n, log_n, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());
        // sync
        gpu::pvMul<<<griddim, blockdim, 0, stream1>>>(n, d_s, d_p, d_q);
        checkCudaErrors(cudaPeekAtLastError());
        // ifft
        gpu::fft<<<griddim, blockdim, 0, stream1>>>(n, log_n, (cuComplex *)d_s, true);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));

        // Phase 7: Deallocation
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaFree(d_q));
        checkCudaErrors(cudaFree(d_s));
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaDeviceReset());

        return s;
    }
}