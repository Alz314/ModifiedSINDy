module OLE_Module
    export solve_SINDy
    export OLE

    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, sparsify, solve_SINDy, SINDy_loss
    using Optimization , OptimizationOptimJL
    using LinearAlgebra


    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct OLE <: SINDy_Alg
        x::Vector{AbstractFloat}
        ADType::SciMLBase.AbstractADType
        opt::Optim.AbstractOptimizer
        lb::Vector{AbstractFloat}
        ub::Vector{AbstractFloat}
    end


    # Constructors for convenience
    # ----------------------------------------------------------------------------------
    OLE(x, ADType, opt) = OLE(x, ADType, opt, [0], [0]) # Constructor for no bounds


    # Functions 
    # ----------------------------------------------------------------------------------
    function solve_SINDy(prob::Modified_SINDy_Problem, OLE_params::OLE)
        #prob.Θ = prob.Lib(prob.u, OLE_params.x)
    
        function loss_f(x, p)
            #OLE_params.x = x
            #println("trying ", x)
            prob.active_Ξ = prob.control_Ξ
            #prob.Θ = prob.Lib(prob.u, OLE_params.x)
            prob.Θ = prob.Lib(prob.u, x)
            Ξes, l = sparsify(prob)
            #l -= prob.η * count(prob.active_Ξ .&& prob.param_Ξ)
            #l -= prob.η * count(prob.active_Ξ)
            #println("x = ", x, " loss = ", l, " count = ", count(prob.active_Ξ .&& prob.param_Ξ))

            # manually regressing with manually added parameterized basis terms 
            # activating the parameterized basis terms (active_Ξ OR param_Ξ)
            prob.active_Ξ = prob.active_Ξ .|| prob.param_Ξ
            n_state=size(prob.du,2)
            if prob.STRRidge 
                for ind=1:n_state
                    biginds = prob.active_Ξ[:,ind]
                    M = transpose(prob.Θ[:,biginds])*prob.Θ[:,biginds]
                    Ξes[biginds,ind] = inv(M + prob.ρ*Diagonal(ones(size(M,1))))*(transpose(prob.Θ[:,biginds])*prob.du[:,ind])
                end
            else
                for ind=1:n_state
                    biginds = prob.active_Ξ[:,ind]
                    Ξes[biginds,ind] = prob.Θ[:,biginds]\prob.du[:,ind]
                end
            end
            l = sum(abs.(prob.Θ * Ξes - prob.du))
            #l = SINDy_loss(prob.du, prob.Θ, Ξes, prob.η)
            return l
        end

        #optf = OptimizationFunction(loss_f, OLE_params.ADType)

        # 
        if OLE_params.lb == [0] && OLE_params.ub == [0]
            sol = optimize(x -> loss_f(x, []), OLE_params.x, OLE_params.opt, Optim.Options(iterations = 50, x_tol = 1e-4, store_trace = true, extended_trace = true); autodiff = :finite)
            #optprob = OptimizationProblem(optf, OLE_params.x, [])
            #optprob = OptimizationProblem(loss, OLE_params.x, [])
        else
            sol = optimize(x -> loss_f(x, []), OLE_params.lb[1], OLE_params.ub[1], OLE_params.x, Fminbox(OLE_params.opt); autodiff = :forward)
            #optprob = OptimizationProblem(optf, OLE_params.x, [], lb = OLE_params.lb, ub = OLE_params.ub)
            #optprob = OptimizationProblem(loss, OLE_params.x, [], lb = OLE_params.lb, ub = OLE_params.ub)
        end
        

        ls = []
        for i in 0.01:0.005:1.5
            push!(ls, loss_f([i], []))
        end

        #sol = solve(optprob, OLE_params.opt)
        #OLE_params.x = sol.u

        #OLE_params.x = solve(optprob, OLE_params.opt).u
        OLE_params.x = sol.minimizer
        
        prob.Θ = prob.Lib(prob.u, OLE_params.x)
        prob.active_Ξ = prob.control_Ξ
        return sparsify(prob)..., sol, ls
    end
end