#include <cassert>
#include <cstdlib>
#include <stdio.h>

#include <cuComplex.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "cpu/cpu.cuh"
#include "gpu/add.cu"
#include "gpu/gpu.cuh"
#include "utils.cuh"
#include "verify.cu"

#define DEBUG true

int main() {
  // output device info and transfer size
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  printf("\nDevice: %s (device memory: %.1f GiB)\n", prop.name,
         prop.totalGlobalMem / double(1 << 30));

  // Phase 0: Sizing
  const unsigned int log_n = Z_LOG_N;
  const unsigned int nElements = Z_N;
  ASSERT_MSG(is_power_of_two(nElements), "FFT only works for powers of two.");
  const size_t nBytes = nElements * sizeof(cuComplex);
  // printf("poly: %i", sizeof(Poly));
  ASSERT_MSG(sizeof(cuComplex) == 2 * sizeof(float), "size of complex");

  printf("Estimated GPU memory requirement size (GiB): %.1f\n",
         nBytes * 3.0 / double(1 << 30));

  // bool verified = gpu::add();

  // std::exit(verified ? EXIT_SUCCESS : EXIT_FAILURE);
}
