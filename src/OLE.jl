module OLE_Module
    export solve_SINDy
    export OLE

    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, sparsify, solve_SINDy
    using Optimization , OptimizationOptimJL


    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct OLE <: SINDy_Alg
        x::Vector{Float64}
        ADType::SciMLBase.AbstractADType
        opt::Optim.AbstractOptimizer
        lb::Vector{Float64}
        ub::Vector{Float64}
    end


    # Constructors for convenience
    # ----------------------------------------------------------------------------------
    OLE(x, ADType, opt) = OLE(x, ADType, opt, [0], [0]) # Constructor for no bounds


    # Functions 
    # ----------------------------------------------------------------------------------
    function solve_SINDy(prob::Modified_SINDy_Problem, OLE_params::OLE)
        prob.Θ = prob.Lib(prob.u, OLE_params.x)
        counter = 0
    
        function loss(x, p)
            prob.Θ = prob.Lib(prob.u, OLE_params.x)
            Ξes, loss = sparsify(prob)
            counter += 1
            return loss
        end

        optf = OptimizationFunction(loss, OLE_params.ADType)

        # 
        if OLE_params.lb == [0] && OLE_params.ub == [0]
            optprob = OptimizationProblem(optf, OLE_params.x)
        else
            optprob = OptimizationProblem(optf, OLE_params.x, lb = OLE_params.lb, ub = OLE_params.ub)
        end

        OLE_params.x = solve(optprob, OLE_params.opt).u

        println(OLE_params.x)
        
        prob.Θ = prob.Lib(prob.u, OLE_params.x)
        return sparsify(prob)..., OLE_params.x, counter
    end
end