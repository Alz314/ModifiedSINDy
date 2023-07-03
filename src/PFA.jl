module PFA_Module
    export PFA 
    export solve_SINDy

    # import functions from base.jl
    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, solve_SINDy, sparsify, lsq_regress

    using DifferentialEquations, ForwardDiff, LinearAlgebra, Distributed
    using Optimization , OptimizationOptimJL

    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct PFA <: SINDy_Alg
        x::Vector{AbstractFloat} # parameters for parameterized basis terms
        cs::AbstractVector{AbstractFloat} 
        train_pct::AbstractFloat
        parallel::Bool # run in parallel (does not work yet)
        option::Int
    end
    
    # Constructors for convenience
    # ----------------------------------------------------------------------------------
    PFA(x, cs, train_pct; parallel = false, option = 1) = PFA(x, cs, train_pct, parallel, option)
    PFA(cs, train_pct; parallel = false, option = 1) = PFA([], cs, train_pct, parallel, option)


    # Functions
    # ----------------------------------------------------------------------------------

    numterms(Ξ) = sum(abs.(Ξ) .> 0)

    
    # defines a function for getting the trajectory of the system (not used anymore)
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

    # defines a function for optimizing the parameters in the parameterized basis terms for a given value of c
    function opt_param_basis_terms!(prob::Modified_SINDy_Problem, PFA_params::PFA)
        function loss_f(x, p)
            prob.active_Ξ = prob.control_Ξ
            prob.Θ = prob.Lib(prob.u, x)
            Ξes, _ = sparsify(prob)
            # activating the parameterized basis terms and regressing once more
            prob.active_Ξ = prob.active_Ξ .|| prob.param_Ξ
            Ξes = lsq_regress(prob)
            l = sum(abs.(prob.Θ * Ξes - prob.du))
            return l
        end

        opt = ParticleSwarm(lower = [0.], upper = [2.0], n_particles = 3)
        sol = optimize(x -> loss_f(x, []), Vector{Float64}(PFA_params.x), opt, Optim.Options(iterations = 50))

        PFA_params.x = sol.minimizer
        prob.Θ = prob.Lib(prob.u, PFA_params.x)
    end


    """
        solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)

    This function solves the SINDy problem using the PFA algorithm.

    # Arguments
        * `prob`: Modified_SINDy_Problem type
        * `PFA_params`: PFA type
    """
    function solve_SINDy(prob::Modified_SINDy_Problem, PFA_params::PFA)
        # Make a train test split
        trainind = Int(ceil(size(prob.Θ, 1)*PFA_params.train_pct)) # index that splits training and test data
        θ_train = prob.Θ[1:trainind, :]
        θ_test = prob.Θ[trainind+1:end, :]
        du_train = prob.du[1:trainind, :]
        du_test = prob.du[trainind+1:end, :]
        u_train = prob.u[1:trainind, :]
        u_test = prob.u[trainind+1:end, :]
        # update Modified_SINDy_Problem object
        prob.u = u_train
        prob.du = du_train
        prob.Θ = θ_train

        Θcond = cond(prob.Θ)

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
                prob.η = c * Θcond
                # update unknown parameterized basis terms
                PFA_params.x != [] && opt_param_basis_terms!(prob, PFA_params)
                # solve for Ξ
                prob.active_Ξ = prob.control_Ξ
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
            prob.η = cbest * Θcond
            # solve for Ξ
            prob.active_Ξ = prob.control_Ξ
            bestΞ, _ = sparsify(prob)
        end
        if PFA_params.parallel != true
            # no parallelization; run sequentially until a bad c value is found
            for c in PFA_params.cs
                # update prob parameters with new value of c
                prob.η = c * Θcond
                # update unknown parameterized basis terms
                PFA_params.x != [] && opt_param_basis_terms!(prob, PFA_params)
                # solve for Ξ
                prob.active_Ξ = prob.control_Ξ
                Ξes, _ = sparsify(prob)
                if count(prob.active_Ξ) == 0
                    println(c)
                    break
                end

    
                # uncomment to calculate trajectory in loss
                #traj = get_trajectory(prob, PFA_params, Ξes)
                #traj_test = traj[trainind+1:end, :]
                #loss = count(prob.active_Ξ) * ( PFA_params.bias * sum(abs2, u_test .- traj_test) + sum(abs2, du_test .- θ_test * Ξes))
                
                
                # get loss
                if PFA_params.option == 1
                    # standard loss
                    loss = count(prob.active_Ξ) * (sum(abs2, du_test .- θ_test * Ξes))
                elseif PFA_params.option == 2
                    # log loss
                    loss = log(count(prob.active_Ξ)) * (sum(abs2, du_test .- θ_test * Ξes))
                elseif PFA_params.option == 3  
                    # AIC loss 
                    nobs_test = size(du_test,1)
                    k = count(prob.active_Ξ)
                    loss = 2*k + nobs_test*log(sum(abs2, du_test .- θ_test*Ξes)/nobs_test) + 2*k*(k+1)/(nobs_test-k-1)
                end
                loss_vals = [loss_vals; loss]
    
                # update best loss
                if loss < minloss
                    minloss = loss
                    cbest = c
                    bestΞ = Ξes
                end
            end
        end

        # update Modified_SINDy_Problem object's parameters with best value of c
        prob.η = cbest * Θcond

        # merge train test data back together in Modified_SINDy_Problem object
        prob.u = [u_train; u_test]
        prob.du = [du_train; du_test]
        prob.Θ = [θ_train; θ_test]

        return bestΞ, minloss, cbest, loss_vals
    end
end