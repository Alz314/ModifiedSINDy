module ModifiedSINDy

export Modified_SINDy_Problem, SINDy_Alg, Default_SINDy, AbstractBasisTerm, BasisTerm, BandedBasisTerm # types
export SINDy_Problem # constructors
export solve_SINDy, ensemble_solve_SINDy, sparsify, CalDerivative # functions
export SG_smoothing, SG_smoothing_optim, smooth_with_kernel # smoothing functions

export OLE
export PFA

#export OLE, ADO, BPSTLSQ

include("base.jl")
include("smoothing.jl")
include("OLE.jl")
include("PFA.jl")
#include("ADO_OLE.jl")
#include("BPSTLSQ.jl")

using .SINDy_Base
using .smoothing
using .OLE_Module
using .PFA_Module
#using .ADO_OLE_Module
#using .BPSTLSQ_Module

end
