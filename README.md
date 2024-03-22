# ZFFT

### Current state

This project is under active development.


### Instructions

To build and run the project, do the following three steps:

**Step 1** First install the necessary OS packages:

Arch

    sudo pacman -S conan cuda cmake

Debian/Ubuntu

    sudo apt install conan cuda cmake

**Step 2** Install the dependencies:

Clone the project

    git clone 'https://github.com/einar-io/zfft'
    cd zfft

Install the dependencies:

    conan install . --build missing

Generate CMake configuration of the parent directory while still in `build/`:

    cd build/
    cmake .. 

This ends the one-time setup. 

**Step 3** Install the dependencies:

Next build the project in `build/`:

    cmake --build .

and run the test suite:

    bin/run_zffttest

For a compined rebuild-run-loop, use the line:

    cmake --build . && bin/run_zffttest


**Clean**

You can clean the build directory by using git:

    git clean -Xid


### Considerations

The challenge is extensive, so I will prioritize finishing one function at a
time, i.e. start with `Add` and write tests, benchmark and profiling.  Then move
on to `Mul`. This way I can demonstrate quality per time unit.

Using complex numbers and not Gaussian integers as the underlying fields implies
some decisions need to be made w.r.t. overflow.

It is inclear whether it is okay to use `cuComplex.h`. I can start out using that
and then migrate it over.

The polynomials are univariate, and we can assume N is known at compile time, so
it suffices to use an array to represent each polynomial, e.g.: P(X) = AX^2 + BX^2 +
C can be represented as:

    P(X) = [ C, B, A, 0, ..., 0 ]
    |P(X)| == N


###### Tuesday

Benchmarks (throughput) makes sense mostly for many polynomials, so we can wait
until we are doing matrices anyway.

The most important achievement is to get a correct implementation of iterative FFT.

It is unclear how much spacial and temporal optimization techniques for matrix
multiplication such as block and register tiling can be applied in this project,
since the element type (N-degree polynomials) do not in genereal fit in
registers.  This is area will not be prioritized.

###### Wednesday

It is clear that we will leave optimizations behind, but we can document them as
we encounter them for later implementation.  The FFT implementation chosen, is
single threaded.  As long as we have enough polynomials, this should not be a
problem.

In the matrix multiplication, each polynomial will be multiplied several times.
Thus, it makes sense to first convert each element (polynomial) into point-value
form and then do the matrix multiplication algorithm and then convert back to
coefficient form.




### Feature Priorities

- [ ] Use Async API
- [ ] End-to-End Tests
- [ ] Benchmarks
- [ ] Visualization
- [ ] (low priority) Wrapping solution in 2nd language:  C++(?), Rust, Zig, Go, Python


### Task

- [x] 1h Sketch a plan
- [x] 4h Read up on polynomail rings in Lauritzen
- [x] Implement `Add`
- [x] Implement iterative FFT
- [x] Write test
- [ ] potentially compare with cuFFT
- [x] Implement Matrix add
- [ ] Benchmark.  Consider using events and timers.
- [ ] Implement Matrix mul
- [ ] Benchmark
- [ ] Report


### Schedule

- Monday: Plan, Study Poly Rings, implement Add e2e, Makefile/conan, Test, simple Benchmark
- Tuesday: Design meaningful benchmark, Read PMPH matmul,
- Wednesday:  implement Mul/FFT
- Thursday:  MatAdd, MatMul
- Friday: MatMul, benchmark
- Saturday: Visualisation, write-up
- Sunday:   Free



### Versions used under development

Only tested Linux platform with Conan2

```bash
$ conan --version
Conan version 2.1.0

$ cmake --version
cmake version 3.28.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```


