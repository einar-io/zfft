#include <vector>
#include "cuComplex.h"
#include "../utils.cuh"
#include <assert.h>
#include "cpu.cuh"

using namespace std;
using namespace cpu;

namespace cpu
{

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

        // fft
        fft(n, log_n, p.data());
        // return p;
        fft(n, log_n, q.data());

        // pvmul
        pvMul(n, s.data(), p.data(), q.data());

        // ifft
        fft(n, log_n, s.data(), true);

        return s;
    }

}