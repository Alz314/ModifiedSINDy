"""
Follow this template for any new algorithms. 
Make sure that you edit Modified_SINDy.jl to have:
    include("alg.jl")
    using .NewModuleName
Also, if you export any additional functions here, you need to export them in Modified_SINDy.jl too
"""

module PFA_Module
    export PFA # need to export new data type
    export solve_SINDy # also export any other functions you would like to be able to call elsewhere

    # import functions from base.jl
    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, solve_SINDy, sparsify

    using DifferentialEquations, ForwardDiff,  LinearAlgebra

    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct PFA <: SINDy_Alg
        x::Vector{Float64}
        tspan::Tuple{Float64, Float64}
        cs::AbstractVector{Float64}
        bias::Float64
        DEAlgorithm::Union{Nothing, DiffEqBase.DEAlgorithm}
        train_pct::Float64
    end


    
    # Constructors for convenience
    # ----------------------------------------------------------------------------------



    # Functions 
    # ----------------------------------------------------------------------------------

    # For new functions, probably best practice to write documentation in this form above the function
    # It's a lot of extra work and not completely necessary, but it would be nice
    """
        solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)

    Explanation of function
    
    # Arguments
        * `prob`: Modified_SINDy_Problem type
        * etc.

    # Notes
        * Optional notes regarding the function 

    # Examples
    ```julia
    julia> # optionally write how this function should be called
    output
    ```
    """

    numterms(Ξ) = sum(abs.(Ξ) .> 0)

    
    # define a function for getting the trajectory of the system
    function get_trajectory(prob::Modified_SINDy_Problem, PFA_params::PFA, Ξ::AbstractMatrix)
        # define the ODE problem
        probODE = ODEProblem((u,p,t)->vec(prob.Lib(reshape(u, 1, 3), PFA_params.x) * Ξ), prob.u[1, :], PFA_params.tspan, [])
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

    using Plots

    function solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)
        sqerr(Ξ) = sum(abs2 , prob.du .- prob.Θ *Ξ)
        #mse(Ξ) = sqrt(sum(abs2 , du .- θ*Ξ))
        #pfloss(Ξ) = numterms(Ξ)*sqerr(Ξ)
        pfloss(Ξ) = numterms(Ξ)*sqerr(Ξ)

        # Make a train test split
        trainind = Int(ceil(size(prob.Θ, 1)*PFA_params.train_pct)) # index that splits training and test data
        θ_train = prob.Θ[1:trainind, :]
        θ_test = prob.Θ[trainind+1:end, :]
        du_train = prob.du[1:trainind, :]
        du_test = prob.du[trainind+1:end, :]
        u_test = prob.u[trainind+1:end, :]
        prob.du = du_train
        prob.Θ = θ_train

        # initial guess
        bestΞ = θ_train \ du_train
        minloss = Inf
        cbest = PFA_params.cs[1]
        loss_vals = []

        for c in PFA_params.cs
            # update prob parameters with new value of c
            prob.η = c * cond(prob.Θ)
            # solve for Ξ
            prob.active_Ξ = ones(size(prob.active_Ξ))


            Ξes, l = sparsify(prob)



            if count(prob.active_Ξ) == 0
                break
            end

            # get loss
            traj = get_trajectory(prob, PFA_params, Ξes)
            traj_test = traj[trainind+1:end, :]
            #loss = count(prob.active_Ξ) * ( PFA_params.bias * sum(abs2, prob.u .- traj) + sum(abs2, du_test .- θ_test * Ξes))
            loss = count(prob.active_Ξ) * ( PFA_params.bias * sum(abs2, u_test .- traj_test) + sum(abs2, du_test .- θ_test * Ξes))
            #loss = count(prob.active_Ξ) * ( sum(abs2, du_test .- θ_test * Ξes))
            loss_vals = [loss_vals; loss]

            # update best loss
            if loss < minloss
                minloss = loss
                cbest = c
                bestΞ = Ξes
            end
        end

        return bestΞ, minloss, cbest, loss_vals
    end
end