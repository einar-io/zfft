#include <stdio.h>
#include "../utils.cuh"
#include "../verify.cu"
#include "gpu.cuh"
#include <vector>
#include <cuComplex.h>

#include "driver_types.h"
#include "../include/helper_cuda.h"

namespace gpu
{
    using namespace std;
    // For performance
    void add_pinned(int n, cuComplex *h_s, cuComplex *h_p, cuComplex *h_q, cuComplex *dh_s, cuComplex *d_p, cuComplex *d_q, cuComplex *d_s)
    {

        // Phase 0: Sizing
        // const unsigned int log_n = log2;
        const unsigned int nElements = n;
        const size_t nBytes = nElements * sizeof(cuComplex);

        // Phase 1: Allocation

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        /*
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
                */

        // Phase 2: Memcpy: Host-to-Device
        checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));
        checkCudaErrors(cudaMemcpyAsync(d_q, h_q, nBytes, cudaMemcpyHostToDevice, stream1));

        // Phase 3: Execute
        // set kernel launch configuration
        dim3 blockdim = dim3(256, 1);
        dim3 griddim = dim3(CEIL_DIV(nElements, blockdim.x), 1);
        add_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));
        checkCudaErrors(cudaStreamDestroy(stream1));
        return;

        /*
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

                return verified;
        */
    }

    // For easing testing
    vector<cuComplex> add_simple(vector<cuComplex> &p, vector<cuComplex> &q)
    //(int n, cuComplex *h_s, cuComplex *h_p, cuComplex *h_q, cuComplex *dh_s, cuComplex *d_p, cuComplex *d_q, cuComplex *d_s)
    {

        // Phase 0: Sizing
        // const unsigned int log_n = log2;
        ASSERT_MSG(p.size() == q.size(), "The two vectors must have same length.");
        const unsigned int nElements = p.size();
        const size_t nBytes = nElements * sizeof(cuComplex);

        vector<cuComplex> s;
        s.resize(p.size());

        // Phase 1: Allocation

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        cuComplex *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

        /*
        checkCudaErrors(cudaMallocHost((void **)&h_p, nBytes));
        checkCudaErrors(cudaMallocHost((void **)&h_q, nBytes));
        checkCudaErrors(cudaMallocHost((void **)&h_s, nBytes));
        checkCudaErrors(cudaMallocHost((void **)&dh_s, nBytes));
        */

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
        dim3 blockdim = dim3(256, 1);
        dim3 griddim = dim3(CEIL_DIV(nElements, blockdim.x), 1);
        add_kernel<<<griddim, blockdim, 0, stream1>>>(nElements, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));

        // Phase 7: Deallocation
        // checkCudaErrors(cudaFreeHost(h_p));
        // checkCudaErrors(cudaFreeHost(h_q));
        // checkCudaErrors(cudaFreeHost(h_s));
        // checkCudaErrors(cudaFreeHost(dh_s));
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaFree(d_q));
        checkCudaErrors(cudaFree(d_s));
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaDeviceReset());

        return s;
    }

}