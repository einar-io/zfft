#ifndef VERIFY_H
#define VERIFY_H

#include "utils.cuh"

__host__ bool verify(const unsigned int n, const cuComplex *h_s, const cuComplex *d_s)
{
    for (int i = 0; i < n; i++)
    {
        bool compare = cuCeqf(h_s[i], d_s[i]);
        ASSERT_MSG(compare, "CPU and GPU results differ.");
        if (!compare)
            return false;
    }
    return true;
}

#endif /* VERIFIY_H */