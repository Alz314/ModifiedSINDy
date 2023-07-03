# ModifiedSINDy  
ModifiedSINDy.jl is a Julia package that implements modified versions of the SINDy (Sparse Identification of Nonlinear Dynamics) algorithm, a popular method for discovering governing equations of dynamical systems from data. This package is still in development and has not yet been added to the Julia Registry.  

## Features  
The ModifiedSINDy package currently provides the following features:  
* Ensemble SINDy: running SINDy on random samples of the data and averaging the results to obtain a more robust model.  
* STLSQ and STRRidge: two different algorithms that can be used as the gradient descent for SINDy.
* ADO SINDy: an algorithm that tries to recover the exact noise of the data while running SINDy. [Described in more detail here](https://github.com/dynamicslab/modified-SINDy)  
* PFA SINDy: uses an optimization scheme to insert parameterized basis terms into the SINDy algorithm.
* A variety of algorithms to perform smoothing and differentiation  
  
## Installation  
To use the ModifiedSINDy package, first clone the repository and then develop the package:  
`using Pkg`  
`Pkg.develop("path/to/ModifiedSINDy")`  
Once you have the package installed, you can import it into your Julia script or notebook:  
```julia
using ModifiedSINDy
```
The package contains a folder of example notebooks demonstrating how to use the ModifiedSINDy package for various systems. (TODO)

## Usage
Define a library of Basis Terms by making of vector of BasisTerm objects:
```julia
using ModifiedSINDy
basis = [BasisTerm(u -> u[:, 1]), 
         BasisTerm(u -> u[:, 1].^3),
         BasisTerm(u -> u[:, 2]),
         BasisTerm(u -> cos.(1.2.*u[:,3])),
         BasisTerm(u -> u[:,3] .* 0 .+ 1)];
```
Then make a SINDy_Alg object to choose an algorithm:  
```julia
alg = PFA(-10:0.1:-1, 0.8)
```

Solve the SINDy problem by using the solve_SINDy function:
```julia
problem = SINDy_Problem(u, du, dt, basis, iter, alg)
solve_SINDy(problem) # use ensemble_solve_SINDy() to run SINDy on different subsets of the data
```

*NOTE: Input matrices for the data should be of shape m by n where m is number of data points and n is dimensions of system
