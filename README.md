# zama-challenge

Zama CUDA Challenge 2024


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


### Feature Priorities

- [ ] Use Async API
- [ ] End-to-End Tests
- [ ] Benchmark Tests
- [ ] Wrapping solution in 2nd language:  C++(?), Rust, Zig, Go, Python

### 



### Task

- [x] 1h Sketch a plan
- [ ] 4h Read up on polynomail rings in Lauritzen
- [ ] Implement `Add`


### Schedule

- Monday: Plan, Study Poly Rings, implement Add e2e, Makefile/conan, Test, simple Benchmark
- Tuesday: Design meaningful benchmark, Read PMPH matmul, implement Mul/FFT
- Wednesday: Matmul
- Thursday: 
- Friday:
- Saturday: Free
- Sunday:   Free


### Requirements

- Only tested Linux platform with Conan

    sudo pacman -S conan cuda

Debian/Ubuntu

   sudo apt install conan 

Then the project is can be started with

    git clone <URL>
    conan install . --build missing

