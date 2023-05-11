module PFA_Module
    export PFA 
    export solve_SINDy

    # import functions from base.jl
    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, solve_SINDy, sparsify

    using DifferentialEquations, ForwardDiff,  LinearAlgebra, Distributed

    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct PFA <: SINDy_Alg
        x::Vector{Float64}
        tspan::Tuple{Float64, Float64}
        cs::AbstractVector{Float64}
        bias::Float64
        DEAlgorithm::Union{Nothing, DiffEqBase.DEAlgorithm}
        train_pct::Float64
        parallel::Bool
    end


    
    # Constructors for convenience
    # ----------------------------------------------------------------------------------
    PFA(x, tspan, cs, bias, DEAlgorithm, train_pct; parallel = false) = PFA(x, tspan, cs, bias, DEAlgorithm, train_pct, parallel)


    # Functions 
    # ----------------------------------------------------------------------------------

    numterms(Ξ) = sum(abs.(Ξ) .> 0)

    
    # define a function for getting the trajectory of the system
    function get_trajectory(prob::Modified_SINDy_Problem, PFA_params::PFA, Ξ::AbstractMatrix)
        # define the ODE problem
        #probODE = ODEProblem((u,p,t)->vec(prob.Lib(reshape(u, 1, 3), PFA_params.x) * Ξ), prob.u[1, :], PFA_params.tspan, [])
        probODE = ODEProblem((u,p,t)->vec(prob.Lib(reshape(u, 1, 3), PFA_params.x) * Ξ), prob.u[1, :], (0.0, prob.dt * (size(prob.u, 1) - 1)), [])
        # solve the ODE problem
        sol = solve(probODE, PFA_params.DEAlgorithm, saveat=prob.dt)
        # get output
        if sol.retcode == ReturnCode.Success
            return Array(sol)'
        else
            #println("Error: ", sol.retcode)
            return Inf * prob.u
        end
    end

    """
        solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)

    This function solves the SINDy problem using the PFA algorithm.

    # Arguments
        * `prob`: Modified_SINDy_Problem type
        * `PFA_params`: PFA type

    # Notes
        * This function is called by Modified_SINDy.jl
    """

    function solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)
        # Make a train test split
        trainind = Int(ceil(size(prob.Θ, 1)*PFA_params.train_pct)) # index that splits training and test data
        θ_train = prob.Θ[1:trainind, :]
        θ_test = prob.Θ[trainind+1:end, :]
        du_train = prob.du[1:trainind, :]
        du_test = prob.du[trainind+1:end, :]
        u_test = prob.u[trainind+1:end, :]
        # update prob parameters with train data
        prob.du = du_train
        prob.Θ = θ_train

        # initial guess
        bestΞ = θ_train \ du_train
        minloss = Inf
        cbest = PFA_params.cs[1]
        loss_vals = []
        if PFA_params.parallel 
            println("parallel")
            loss_vals = zeros(length(PFA_params.cs))
            # parallelize over c values, trying every c value provided and then choosing the minimum loss
            @sync @distributed for i in 1:length(PFA_params.cs)
                c = PFA_params.cs[i]
                # update prob parameters with new value of c
                prob.η = c * cond(prob.Θ)
                # solve for Ξ
                prob.active_Ξ = ones(size(prob.active_Ξ))
                Ξes, _ = sparsify(prob)

                # get loss
                # no trajectory error 
                if count(prob.active_Ξ) == 0
                    loss_vals[i] = Inf
                else
                    loss_vals[i] = count(prob.active_Ξ) * (sum(abs2, du_test .- θ_test * Ξes))
                end
            end
            best_index = argmin(loss_vals)
            minloss = loss_vals[best_index]
            cbest = PFA_params.cs[best_index]
            # update prob parameters with new value of c
            prob.η = cbest * cond(prob.Θ)
            # solve for Ξ
            prob.active_Ξ = ones(size(prob.active_Ξ))
            bestΞ, _ = sparsify(prob)
        end
        if PFA_params.parallel != true
            # no parallelization; run sequentially until a bad c value is found
            for c in PFA_params.cs
                # update prob parameters with new value of c
                prob.η = c * cond(prob.Θ)
                # solve for Ξ
                prob.active_Ξ = ones(size(prob.active_Ξ))
                Ξes, _ = sparsify(prob)
                if count(prob.active_Ξ) == 0
                    println(c)
                    break
                end

                # get loss
    
                # uncomment to calculate trajectory
                #traj = get_trajectory(prob, PFA_params, Ξes)
                #traj_test = traj[trainind+1:end, :]
                #loss = count(prob.active_Ξ) * ( PFA_params.bias * sum(abs2, u_test .- traj_test) + sum(abs2, du_test .- θ_test * Ξes))
                
                # no trajectory error 
                loss = count(prob.active_Ξ) * (sum(abs2, du_test .- θ_test * Ξes))
                loss_vals = [loss_vals; loss]
    
                # update best loss
                if loss < minloss
                    minloss = loss
                    cbest = c
                    bestΞ = Ξes
                end
            end
        end

        return bestΞ, minloss, cbest, loss_vals
    end
end