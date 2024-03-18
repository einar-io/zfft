
#include <cuComplex.h>
#include <vector>
#include "../utils.cuh"

using namespace std;

namespace cpu
{

    // This does not need to be
    void add_pinned(int nElements, cuComplex *h_s, cuComplex *h_p, cuComplex *h_q, cuComplex *dh_s = nullptr, cuComplex *d_p = nullptr, cuComplex *d_q = nullptr, cuComplex *d_s = nullptr)
    {
        for (int i = 0; i < nElements; i++)
        {
            h_s[i] = cuCaddf(h_p[i], h_q[i]);
        }
        return;
    }

    // This is simle
    vector<cuComplex> add_simple(vector<cuComplex> &p, vector<cuComplex> &q)
    {
        ASSERT_MSG(p.size() == q.size(), "The two vectors must have same length.");
        vector<cuComplex> s;
        s.resize(p.size());

        for (int i = 0; i < p.size(); i++)
        {
            s[i] = cuCaddf(p[i], q[i]);
        }

        return s;
    }

}