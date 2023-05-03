# ModifiedSINDy  
ModifiedSINDy.jl is a Julia package that implements modified versions of the SINDy (Sparse Identification of Nonlinear Dynamics) algorithm, a popular method for discovering governing equations of dynamical systems from data. This package is still in development and has not yet been added to the Julia Registry.  

## Features  
The ModifiedSINDy package currently provides the following additional features:  
* Bagging SINDy: running SINDy on random samples of the data and averaging the results to obtain a more robust model.  
* STLSQ and STRRidge: two different algorithms that can be used as the gradient descent for SINDy.  
* ADO SINDy: an algorithm that tries to recover the exact noise of the data while running SINDy. [Described in more detail here](https://github.com/dynamicslab/modified-SINDy)  
* Out of Library Estimation (OLE) SINDy: uses an optimization scheme to insert parameterized basis terms into the SINDy algorithm.  
  
## Usage  
To use the ModifiedSINDy package, first clone the repository and then develop the package:  
`using Pkg`  
`Pkg.develop("path/to/ModifiedSINDy")`  
Once you have the package installed, you can import it into your Julia script or notebook:  
`using ModifiedSINDy`  
The package contains a folder of example notebooks demonstrating how to use the ModifiedSINDy package for various systems.
