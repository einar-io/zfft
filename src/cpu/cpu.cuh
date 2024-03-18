
#ifndef CPU_H
#define CPU_H

#include <cuComplex.h>
#include "../utils.cuh"
#include <vector>

using namespace std;

namespace cpu
{
    __device__ static void reverse_flipshift(uint32_t num, uint32_t lg_n, uint32_t *out)
    {
        // flip+shift __brev
        *out = 0; // misisng isntructionrbit()
        num >>= (CHAR_BIT * sizeof(uint32_t) - lg_n);
        return;
    }

    // https://cp-algorithms.com/algebra/fft.html
    __host__ static void reverse_cp(uint32_t num, uint32_t lg_n, uint32_t *out)
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

    __host__ void permute(int n, int log_n, cuComplex *a)
    {

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

    __host__ void pvMul(int n, cuComplex *dst, const cuComplex *src1, const cuComplex *src2)
    {
        for (int i = 0; i < n; i++)
        {
            dst[i] = cuCmulf(src1[i], src2[i]);
        }
    }

    __host__ void fft(int n, int log_n, cuComplex *a, bool invert = false)
    {
        permute(n, log_n, a);

        for (int len = 2; len <= n; len <<= 1)
        {
            float ang = 2 * PI / len * (invert ? -1 : 1);
            cuComplex wlen = make_cuComplex(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len)
            {
                cuComplex w = make_cuComplex(1, 0);
                for (int j = 0; j < len / 2; j++)
                {
                    cuComplex u = a[i + j];
                    cuComplex v = cuCmulf(a[i + j + len / 2], w);
                    a[i + j] = cuCaddf(u, v);
                    a[i + j + len / 2] = cuCsubf(u, v);
                    w = cuCmulf(wlen, w);
                }
            }
        }

        if (invert)
        {
            cuComplex nC = make_cuComplex(float(n), 0.0f);
            for (int i = 0; i < n; i++)
            {
                a[i] = cuCdivf(a[i], nC);
            }
        }
    }

    __host__ vector<cuComplex> fft_simple(vector<cuComplex> const &p)
    {
        vector<cuComplex> s(p.begin(), p.end());
        cpu::fft(s.size(), log_2(s.size()), s.data());
        return s;
    }

}

#endif /* CPU_H */