
#ifndef GPU_H
#define GPU_H

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuComplex.h>
#include "../utils.cuh"
#include <vector>

#include "driver_types.h"
#include "../include/helper_cuda.h"

namespace gpu
{
    using namespace std;

    __global__ void add_kernel(const int n, cuComplex *dst, const cuComplex *src1, const cuComplex *src2)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (n <= i)
            return;

        dst[i] = cuCaddf(src1[i], src2[i]);
    }

    __global__ void mul_kernel(const int n, const int log_n, cuComplex *dst, cuComplex *src1, cuComplex *src2)
    {
        // int i = blockIdx.x * blockDim.x + threadIdx.x;
        // if (i < n)

        /*
            // pad
            fftGPU(8, 3, (cuComplex *)src1);
            fftGPU(8, 3, (cuComplex *)src2);
            // sync
            __syncthreads();
            pointValueMulGPU(n, dst, src1, src2);
            //// ifft(n, dst);
            fftGPU(8, 3, (cuComplex *)dst, true);
            */
    }
    __device__ static void reverse_flipshift(uint32_t num, uint32_t lg_n, uint32_t *out)
    {
        // flip+shift __brev
        *out = __brev(num) >> (CHAR_BIT * sizeof(uint32_t) - lg_n);
        return;
    }

    // https://cp-algorithms.com/algebra/fft.html
    __device__ static void reverse_cp(uint32_t num, uint32_t lg_n, uint32_t *out)
    {
        int res = 0;
        for (int i = 0; i < lg_n; i++)
        {
            if (num & (1 << i))
                res |= 1 << (lg_n - 1 - i);
        }
        *out = res;
        return;
    }

    __device__ static void permute(int n, int log_n, cuComplex *a)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (n <= i)
            return;

        uint32_t j;
        reverse_flipshift(i, log_n, &j);

        if (i < j)
        {
            cuComplex tmp = a[i];
            __syncthreads();
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    __global__ static void fft(int n, int log_n, cuComplex *a, bool invert = false)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (n <= i)
            return;

        permute(n, log_n, a);

        // partitions from pairs up to full size
        // side-effects are contained within a partition
        for (int len = 2; len <= n; len <<= 1)
        {
            int part_id = i / len;
            int part_offset = i % len;
            int id = part_id * len + part_offset;

            float ang = 2 * PI / len * (invert ? -1 : 1);
            cuComplex wlen = make_cuComplex(cos(ang), sin(ang));

            // in each partition
            // for (int i = 0; i < n; i += len)
            int i = part_id;
            {
                // cuComplex w = make_cuComplex(1, 0);
                cuComplex w = make_cuComplex(1, 0);
                for (int k = 0; k < part_offset; k++)
                {
                    w = cuCmulf(wlen, w);
                }

                __syncthreads();

                // first half of the partition
                const int half = len >> 1;
                // for (int j = 0; j < half; j++)
                int j = part_offset;
                if (j < half)
                {
                    cuComplex u = a[id];
                    cuComplex v = cuCmulf(a[id + half], w);
                    __syncthreads();
                    a[id] = cuCaddf(u, v);
                    a[id + half] = cuCsubf(u, v);
                    // w = cuCmulf(wlen, w);
                }
            }
            __syncthreads();
        }

        if (invert)
        {
            cuComplex nC = make_cuComplex(float(n), 0.0f);
            //  for (int i = 0; i < n; i++)
            {
                a[i] = cuCdivf(a[i], nC);
            }
        }
    }

    __global__ void pvMul(int n, cuComplex *dst, const cuComplex *src1, const cuComplex *src2)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (n <= i)
            return;

        dst[i] = cuCmulf(src1[i], src2[i]);
    }

    __global__ void permuteK2(int n, int log_n, cuComplex *a)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (0 != i)
            return;

        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;

            if (i < j)
            {
                cuComplex tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }

    __global__ static void permuteK(int n, int log_n, cuComplex *a)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (n <= i)
            return;

        uint32_t j;
        reverse_flipshift(i, log_n, &j);

        if (i < j)
        {
            cuComplex tmp = a[i];
            __syncthreads();
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    vector<cuComplex> permute_simple(vector<cuComplex> &p)
    {

        ASSERT_MSG(is_power_of_two(p.size()), "Must be a power of two.");

        int n = p.size();
        int log_n = log_2(n);
        ASSERT_MSG((1 << log_n) == n, "2^log_n = n.");

        // Phase 0: Sizing
        int nElements = n;
        int nBytes = n * sizeof(cuComplex);

        // Phase 1: Allocation

        vector<cuComplex> s;
        s.resize(p.size());

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        cuComplex *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

        h_p = p.data();
        dh_s = s.data();

        checkCudaErrors(cudaMalloc((void **)&d_p, nBytes));

        // memset
        checkCudaErrors(cudaMemset((void *)d_p, 0, nBytes));

        // Phase 2: Memcpy: Host-to-Device
        checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));

        // Phase 3: Execute
        // set kernel launch configuration

        dim3 blockdim = dim3(32, 1);
        dim3 griddim = dim3(CEIL_DIV(32, blockdim.x), 1);

        gpu::permuteK<<<griddim, blockdim, 0, stream1>>>(n, log_n, (cuComplex *)d_p);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_p, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));

        // Phase 7: Deallocation
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaDeviceReset());

        return s;
    }

    vector<cuComplex> fft_simple(const vector<cuComplex> &p_orig, bool invert = false)
    {
        vector<cuComplex> p(p_orig.begin(), p_orig.end());

        ASSERT_MSG(is_power_of_two(p.size()), "Must be a power of two.");

        int n = p.size();
        int log_n = log_2(n);
        ASSERT_MSG((1 << log_n) == n, "2^log_n = n.");

        // Phase 0: Sizing
        int nElements = n;
        int nBytes = n * sizeof(cuComplex);

        // Phase 1: Allocation

        vector<cuComplex> s;
        s.resize(p.size());

        // Create non-default stream for Async
        cudaStream_t stream1;
        checkCudaErrors(cudaStreamCreate(&stream1));

        cuComplex *h_p, *h_q, *h_s, *d_p, *d_q, *d_s, *dh_s;

        h_p = p.data();
        dh_s = s.data();

        checkCudaErrors(cudaMalloc((void **)&d_p, nBytes));

        // memset
        checkCudaErrors(cudaMemset((void *)d_p, 0, nBytes));

        // Phase 2: Memcpy: Host-to-Device
        checkCudaErrors(cudaMemcpyAsync(d_p, h_p, nBytes, cudaMemcpyHostToDevice, stream1));

        // Phase 3: Execute
        // set kernel launch configuration

        dim3 blockdim = dim3(32, 1);
        dim3 griddim = dim3(CEIL_DIV(32, blockdim.x), 1);

        gpu::fft<<<griddim, blockdim, 0, stream1>>>(n, log_n, (cuComplex *)d_p, invert);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_p, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));

        // Phase 7: Deallocation
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaDeviceReset());

        return s;
    }

    vector<cuComplex> pvMul_simple(vector<cuComplex> &p, vector<cuComplex> &q)
    {

        ASSERT_MSG(is_power_of_two(p.size()), "Must be a power of two.");
        ASSERT_MSG(p.size() == q.size(), "The two vectors must have same length.");

        int n = p.size();
        int log_n = log_2(n);
        ASSERT_MSG((1 << log_n) == n, "2^log_n = n.");

        // Phase 0: Sizing
        int nElements = n;
        int nBytes = n * sizeof(cuComplex);

        // Phase 1: Allocation

        vector<cuComplex> s;
        s.resize(p.size());

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

        gpu::pvMul<<<griddim, blockdim, 0, stream1>>>(n, (cuComplex *)d_s, (cuComplex *)d_p, (cuComplex *)d_q);
        checkCudaErrors(cudaPeekAtLastError());

        // Phase 4: Memcpy: Device-to-Host
        checkCudaErrors(cudaMemcpyAsync(dh_s, d_s, nBytes, cudaMemcpyDeviceToHost, stream1));

        checkCudaErrors(cudaStreamSynchronize(stream1));

        // Phase 7: Deallocation
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaDeviceReset());

        return s;
    }

}

#endif /* GPU_H */