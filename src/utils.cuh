
#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <assert.h>

#define Z_LOG_N 5
#define Z_N (1 << (Z_LOG_N))
// #define N (1 << 26)
#define DEBUG true

#define CEIL_DIV(x, y) (((x) + ((y)-1)) / (y))
#define ASSERT_MSG(exp, msg) assert(((void)msg, exp))

const float PI = CUDART_PI_F;

__host__ __device__ static __inline__ bool cuCeqf(cuFloatComplex x,
                                                  cuFloatComplex y)
{
    // todo: better float comparison or report max error
    return cuCrealf(x) == cuCrealf(y) && cuCimagf(x) == cuCimagf(y);
}

// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
__host__ static int cpu_next_power_of_2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    v += (v == 0);

    return v;
}

// https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
__host__ bool is_power_of_two(const unsigned int v)
{
    return v && !(v & (v - 1));
}

// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious
__host__ unsigned int log_2(unsigned int v) // 32-bit word to find the log base 2 of
{
    unsigned int r = 0; // r will be lg(v)
    while (v >>= 1)     // unroll for more speed...
    {
        r++;
    }
    return r;
}

#endif /* UTILS_H */
