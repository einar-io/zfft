#include <cuComplex.h>
#include <boost/math/special_functions/next.hpp>
#include <gtest/gtest.h>
#include <helper_cuda.h>
#include "../src/cpu/add.cu"
#include "../src/gpu/add.cu"
#include "../src/cpu/mul.cu"
#include "../src/cpu/cpu.cuh"
#include "../src/gpu/mul.cu"
#include "../src/utils.cuh"
#include <vector>
#include <float.h>

using namespace std;

#define MAXDIFF (10E-5)

// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
#define EXPECT_COMPLEX_EQ(CPU, GPU)                                                                        \
  do                                                                                                       \
  {                                                                                                        \
    float cpu_r = cuCrealf(CPU);                                                                           \
    float cpu_i = cuCimagf(CPU);                                                                           \
    float gpu_r = cuCrealf(GPU);                                                                           \
    float gpu_i = cuCimagf(GPU);                                                                           \
    float diff_r = fabs(cpu_r - gpu_r);                                                                    \
    float diff_i = fabs(cpu_i - gpu_i);                                                                    \
    if (diff_r <= MAXDIFF && diff_i <= MAXDIFF)                                                            \
      continue;                                                                                            \
    float max_r = max(cpu_r, gpu_r);                                                                       \
    float max_i = max(cpu_i, gpu_i);                                                                       \
    EXPECT_NEAR(cpu_r, gpu_r, FLT_EPSILON *max_r) << "ULP: " << boost::math::float_distance(cpu_r, gpu_r); \
    EXPECT_NEAR(cpu_i, gpu_i, FLT_EPSILON *max_i) << "ULP: " << boost::math::float_distance(cpu_i, gpu_i); \
  } while (0)

TEST(meta, macro_EXPECT_COMPLEX_EQ)
{
  auto c = make_cuComplex(float(1), float(-1));
  auto d = make_cuComplex(float(1), float(1));
  EXPECT_COMPLEX_EQ(c, cuConjf(d));
}

TEST(meta, int_coersion)
{
  cuComplex z = make_cuComplex(0.0f, 0.0f);
  cuComplex mz = make_cuComplex(0.0f, -0.0f);
  cuComplex m = make_cuComplex(1.0f, -1.0f);
  cuComplex tf = make_cuComplex(42.f, 42.0f);
  vector<cuComplex> p = {z, mz, m, tf};
  vector<cuComplex> q = {{0, 0}, {0, -0}, {1, -1}, {42, 42}};
  for (int i = 0; i < p.size(); i++)
  {
    EXPECT_COMPLEX_EQ(p[i], q[i]);
  }
}

TEST(add, add_simple)
{
  using namespace std;

  vector<cuComplex> p = {{2, 3}, {5, 7}, {11, 13}, {17, 19}};
  vector<cuComplex> q = {{2, -3}, {5, -7}, {11, -13}, {17, -19}};

  vector<cuComplex> cpu_sum = cpu::add_simple(p, q);
  vector<cuComplex> gpu_sum = gpu::add_simple(p, q);

  for (int i = 0; i < p.size(); i++)
  {
    EXPECT_COMPLEX_EQ(cpu_sum[i], gpu_sum[i]);
  }
}

TEST(add, add_pinned)
{
  //  Phase 0: Sizing
  const unsigned int log_n = 10;
  const unsigned int nElements = 1 << log_n;
  const size_t nBytes = nElements * sizeof(cuComplex);

  // Phase 1: Allocation

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
  }

  cpu::add_pinned(nElements, h_s, h_p, h_q);
  gpu::add_pinned(nElements, nullptr, h_p, h_q, dh_s, d_p, d_q, d_s);

  for (int i = 0; i < nElements; i++)
  {
    EXPECT_COMPLEX_EQ(dh_s[i], h_s[i]);
  }

  // Phase 7: Deallocation
  checkCudaErrors(cudaFreeHost(h_p));
  checkCudaErrors(cudaFreeHost(h_q));
  checkCudaErrors(cudaFreeHost(h_s));
  checkCudaErrors(cudaFreeHost(dh_s));
  checkCudaErrors(cudaFree(d_p));
  checkCudaErrors(cudaFree(d_q));
  checkCudaErrors(cudaFree(d_s));
  checkCudaErrors(cudaDeviceReset());
}

TEST(mul, cpu_permute)
{
  vector<cuComplex> p = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};
  vector<cuComplex> s = {{0, 0}, {4, 0}, {2, 0}, {6, 0}, {1, 0}, {5, 0}, {3, 0}, {7, 0}};

  cpu::permute(p.size(), log_2(p.size()), p.data());

  for (int i = 0; i < s.size(); i++)
  {
    EXPECT_COMPLEX_EQ(p[i], s[i]);
  }
}

TEST(mul, gpu_permute)
{
  vector<cuComplex> p = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};
  vector<cuComplex> s = {{0, 0}, {4, 0}, {2, 0}, {6, 0}, {1, 0}, {5, 0}, {3, 0}, {7, 0}};

  vector<cuComplex> g = gpu::permute_simple(p);

  for (int i = 0; i < s.size(); i++)
  {
    EXPECT_COMPLEX_EQ(g[i], s[i]);
  }
}

TEST(mul, mul_simple_known_values)
{
  // (1x^2 + 1x + 1) * (1x^2 + 1x + 1) = 1x^4 + 2x^3 + 3x^2 + 2x + 1
  // [1,1,1] * [1,1,1] = [1,2,3,2,1]
  cuComplex z = make_cuComplex(0.0f, 0.0f);
  cuComplex o = make_cuComplex(1.0f, 0.0f);
  vector<cuComplex> p = {o, o, o, z};
  vector<cuComplex> q = {o, o, o, z};
  vector<cuComplex> s = {{1, 0}, {2, 0}, {3, 0}, {2, 0}, {1, 0}, z, z, z};
  // vector<float> s = {1, 2, 3, 2, 1, 0, 0, 0};

  vector<cuComplex> cpu_sum = cpu::mul_simple(p, q);
  // vector<cuComplex> gpu_sum = gpu::mul_simple(p, q);
  // vector<cuComplex> gpu_sum(p.size() * 2);

  for (int i = 0; i < s.size(); i++)
  {
    // printf("cpu_sum[%i]: %f %f \n", i, cpu_sum[i].x, cpu_sum[i].y);
    //  EXPECT_EQ((cuCrealf(cpu_sum[i])), s[i]);
    EXPECT_COMPLEX_EQ(cpu_sum[i], s[i]);
  }
}

TEST(mul, fft_two)
{
  vector<cuComplex> c = {{3, 5}, {7, 11}};
  vector<cuComplex> g = {{3, 5}, {7, 11}};
  int n = c.size(), log_n = log_2(n);

  cpu::fft(n, log_n, c.data());
  auto gout = gpu::fft_simple(g);

  for (int i = 0; i < n; i++)
  {
    EXPECT_COMPLEX_EQ(c[i], gout[i]);
  }
}

TEST(mul, fft_four)
{
  vector<cuComplex> c = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  vector<cuComplex> g = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  int n = c.size(), log_n = log_2(n);

  cpu::fft(n, log_n, c.data());
  vector<cuComplex> go = gpu::fft_simple(g);

  for (int i = 0; i < n; i++)
  {
    EXPECT_COMPLEX_EQ(c[i], go[i]);
  }
}

TEST(mul, fft_eight)
{
  vector<cuComplex> c = {{8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
  vector<cuComplex> g = {{8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
  int n = c.size(), log_n = log_2(n);

  cpu::fft(n, log_n, c.data());
  vector<cuComplex> go = gpu::fft_simple(g);

  for (int i = 0; i < n; i++)
  {
    EXPECT_COMPLEX_EQ(c[i], go[i]);
  }
}

TEST(mul, ifft_eight)
{
  vector<cuComplex> c = {{8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
  vector<cuComplex> g = {{8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
  int n = c.size(), log_n = log_2(n);

  cpu::fft(n, log_n, c.data(), true);
  vector<cuComplex> go = gpu::fft_simple(g, true);

  for (int i = 0; i < n; i++)
  {
    EXPECT_COMPLEX_EQ(c[i], go[i]);
  }
}

TEST(mul, mul_simple)
{
  vector<cuComplex> p = {{2, 3}, {5, 7}, {11, 13}, {17, 19}};
  vector<cuComplex> q = {{2, -3}, {5, -7}, {11, -13}, {17, -19}};

  vector<cuComplex> cpu_sum = cpu::mul_simple(p, q);
  vector<cuComplex> gpu_sum = gpu::mul_simple(p, q);

  for (int i = 0; i < p.size(); i++)
  {
    EXPECT_COMPLEX_EQ(cpu_sum[i], gpu_sum[i]);
  }
}

TEST(mul, mul_simple_human_readable)
{
  const vector<cuComplex> p = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  const vector<cuComplex> q = {{2, 0}, {2, 0}, {2, 0}, {2, 0}};

  vector<cuComplex> cp = cpu::fft_simple(p);
  vector<cuComplex> cq = cpu::fft_simple(q);
  vector<cuComplex> gp = gpu::fft_simple(p);
  vector<cuComplex> gq = gpu::fft_simple(q);

  EXPECT_EQ(cp.size(), cq.size());
  EXPECT_EQ(cq.size(), gp.size());
  EXPECT_EQ(gp.size(), gq.size());
  EXPECT_EQ(gq.size(), cp.size());

  for (int i = 0; i < cp.size(); i++)
  {
    EXPECT_COMPLEX_EQ(cp[i], gp[i]);
    EXPECT_COMPLEX_EQ(cq[i], gq[i]);
  }

  vector<cuComplex> c = cpu::mul_simple(p, q);
  vector<cuComplex> g = gpu::mul_simple(p, q);

  EXPECT_EQ(8, g.size());
  EXPECT_EQ(c.size(), g.size());
  for (int i = 0; i < c.size(); i++)
  {
    EXPECT_COMPLEX_EQ(c[i], g[i]);
  }
}

TEST(mul, pvmul)
{
  vector<cuComplex> p = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  vector<cuComplex> q = {{2, 0}, {2, 0}, {2, 0}, {2, 0}};

  vector<cuComplex> g = gpu::pvMul_simple(p, q);
  cpu::pvMul(p.size(), p.data(), p.data(), q.data());

  for (int i = 0; i < p.size(); i++)
  {
    // printf("p[%i]: %f %f\n", i, p[i].x, p[i].y);
    //  EXPECT_COMPLEX_EQ(c[i], g[i]);
  }
  for (int i = 0; i < p.size(); i++)
  {
    // printf("p[%i]: %f %f\n", i, g[i].x, g[i].y);
    EXPECT_COMPLEX_EQ(p[i], g[i]);
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}