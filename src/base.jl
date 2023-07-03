module SINDy_Base
    export Modified_SINDy_Problem, SINDy_Alg, Default_SINDy, AbstractBasisTerm, BasisTerm, BandedBasisTerm # types
    export SINDy_Problem # constructors
    export solve_SINDy, ensemble_solve_SINDy, sparsify, CalDerivative, SINDy_loss, _is_converged, get_trajectory, calc_AIC, calc_RMSE # functions

    using LinearAlgebra, StatsBase
    using DifferentialEquations

    
    # New Data types
    # ----------------------------------------------------------------------------------
    abstract type SINDy_Alg end
    abstract type AbstractBasisTerm end

    struct Default_SINDy <: SINDy_Alg 
        x::Vector{AbstractFloat}
    end

    mutable struct BasisTerm <:AbstractBasisTerm
        theta::Function
        parameterized::Bool
    end

    mutable struct BandedBasisTerm <:AbstractBasisTerm
        theta::Function
        upper_bound::AbstractFloat
        lower_bound::AbstractFloat
        parameterized::Bool
    end

    mutable struct Modified_SINDy_Problem
        u::Matrix{AbstractFloat}
        du::Matrix{AbstractFloat}
        dt::AbstractFloat
        basis::AbstractArray{<: AbstractBasisTerm}
        Lib::Function
        alg::SINDy_Alg
        STRRidge::Bool
        λs::Vector{AbstractFloat}
        upper_bounds::Vector{AbstractFloat}
        lower_bounds::Vector{AbstractFloat}
        iter::Int16
        ρ::AbstractFloat
        η::AbstractFloat
        abstol::AbstractFloat
        reltol::AbstractFloat
        Θ::Matrix{AbstractFloat}
        active_Ξ::Matrix{Bool}
        param_Ξ::Matrix{Bool}
        control_Ξ::Matrix{Bool}
    end

    # Constructors for convenience
    # ----------------------------------------------------------------------------------
    function SINDy_Problem(u, du, dt, basis, iter, alg; STRRidge = false, λs = Vector{AbstractFloat}([]), control = []) 
        Lib = generateLibrary(basis)
        ubs, lbs = determineBounds(basis)
        #ubs = repeat(ubs, 1, size(u)[2])
        #lbs = repeat(lbs, 1, size(u)[2])

        active_Ξ = ones(Bool, (length(basis), size(u)[2]))
        param_Ξ = zeros(Bool, size(active_Ξ))
        for i in 1:length(basis)
            if basis[i].parameterized
                param_Ξ[i,:] .= true
            end
        end

        if control == []
            control = ones(Bool, size(active_Ξ))
        elseif size(control) != size(active_Ξ)
            controlsize = size(control)
            activesize = size(active_Ξ)
            error("Dimensions of control matrix do not match dimensions of basis. Control matrix has dimensions $controlsize, basis has dimensions $activesize")
        end

        Θ = Lib(u, alg.x)
        #ρ = mean(log10.(eigen(Θ'*Θ).values))
        ρ = mean(log10.(abs.(eigen(Θ'*Θ).values)))
        η = ρ
        return Modified_SINDy_Problem(u, du, dt, basis, Lib, alg, STRRidge, λs, ubs, lbs, iter, ρ, η, 0.000001, 0.000001, Θ, active_Ξ, param_Ξ, control)
    end

    BasisTerm(theta::Function) = BasisTerm(theta, false)

    BasisTerm(theta::Function, lb, ub) = BandedBasisTerm(theta, lb, ub, false)

    SINDy_Problem(u, du, dt, basis, iter; STRRidge=false, λs = [], control = []) = SINDy_Problem(u, du, dt, basis, iter, Default_SINDy(Vector{AbstractFloat}()); STRRidge = STRRidge, λs = λs, control = control)

    # Functions 
    # ----------------------------------------------------------------------------------
    function generateLibrary(basis::AbstractArray{<: AbstractBasisTerm})
        n = length(basis)
        library = Vector{Function}(undef, n)

        for i in 1:n
            if basis[i].parameterized
                library[i] = basis[i].theta
            else
                new_theta(u, x) = basis[i].theta(u)
                library[i] = new_theta
            end
        end

        function Lib(u, x)
            return reduce(hcat,[theta(u, x) for theta in library])
        end
        return Lib
    end

    function determineBounds(basis::AbstractArray{<: AbstractBasisTerm})
        n = length(basis)
        upper_bounds = zeros(n)
        lower_bounds = zeros(n)
        for i in 1:n
            if basis[i] isa BandedBasisTerm
                upper_bounds[i] = basis[i].upper_bound
                lower_bounds[i] = basis[i].lower_bound
            else
                upper_bounds[i] = Inf
                lower_bounds[i] = -Inf
            end
        end
        
        return upper_bounds, lower_bounds
    end

    function solve_SINDy(prob::Modified_SINDy_Problem)
        # Takes in any Modified_SINDy_Problem and calls the correct function to solve the problem

        if prob.alg isa Default_SINDy
            #prob.Θ = prob.Lib(prob.u) # for default SINDy, you don't need to input a Θ
            return sparsify(prob)
        else
            return solve_SINDy(prob, prob.alg) # call the corresponding solve function for the algorithm type
        end
        return 0
    end

    SINDy_loss(du, Θ, Ξes, ρ) = sum(abs.(Θ * Ξes - du)) + ρ*count(x-> abs.(x)>0, Ξes)

    function _is_converged(X, X_prev, abstol, reltol)::Bool
        Δ = norm(X .- X_prev)
        Δ < abstol && return true
        δ = Δ / norm(X)
        δ < reltol && return true
        return false
    end

    function generate_lambdas(coeffs)
        # generates a new set of lambdas for the next iteration of sparse regression
        # coeffs is a vector or matrix of coefficients from the previous iteration
        # returns a vector of lambdas for the next iteration
        
        # sort our current coefficients
        coeffs = abs.(vec(coeffs))
        coeffs = sort(coeffs[coeffs .> 0])
        # use a tolerance to determine which coefficients are about the same
        # this is to avoid having too many lambda values
        # right now I just take the mean of the difference between consecutive coefficients
        tol = median(coeffs[2:end] - coeffs[1:end-1])
        # initialize our new lambdas with the first coefficient plus the tolerance
        λs = [coeffs[1] + tol]
        # loop through the rest of the coefficients
        for i in 2:length(coeffs)
            # if the difference between the current coefficient and the last lambda is greater than the tolerance
            if coeffs[i] - λs[end] > tol
                # add the current coefficient to the list of lambdas
                push!(λs, coeffs[i])
            end
        end
        # remove last lambda so that we won't have an empty set of coefficients after sparsifying
        pop!(λs)
        return λs  
    end

    # define a function to perform least squares regression with an input SINDy problem
    function lsq_regress(prob::Modified_SINDy_Problem, ρ = nothing)
        # if no ρ is given, use the one defined in the SINDy problem
        if ρ == nothing
            ρ = prob.ρ
        end

        Ξes = zeros(AbstractFloat, size(prob.active_Ξ))

        if prob.STRRidge 
            for ind=1:size(prob.du,2)
                biginds = prob.active_Ξ[:,ind]
                M = Matrix{Float64}(transpose(prob.Θ[:,biginds])*prob.Θ[:,biginds])
                #display(M)
                #display(temp_Ξes)
                Ξes[biginds,ind] .= inv(M + prob.ρ*Diagonal(ones(size(M,1))))*(transpose(prob.Θ[:,biginds])*prob.du[:,ind])
            end
        else
            for ind=1:size(prob.du,2)
                biginds = prob.active_Ξ[:,ind]
                Ξes[biginds,ind] .= prob.Θ[:,biginds]\prob.du[:,ind]
            end
        end
        return Ξes
    end

    function get_trajectory(prob::Modified_SINDy_Problem, Ξ::AbstractMatrix, DEAlgorithm = Tsit5())
        # define the ODE problem
        #probODE = ODEProblem((u,p,t)->vec(prob.Lib(reshape(u, 1, 3), PFA_params.x) * Ξ), prob.u[1, :], PFA_params.tspan, [])
        probODE = ODEProblem((u,p,t)->Vector{Float64}(vec(prob.Lib(reshape(u, 1, size(prob.u, 2)), prob.alg.x) * Ξ)), Vector{Float64}(prob.u[1, :]), (Float64(0.0), Float64(prob.dt * (size(prob.u, 1) - 1))), [])
        # solve the ODE problem
        sol = solve(probODE, DEAlgorithm, saveat=prob.dt)
        #sol = solve(probODE, DEAlgorithm, saveat=prob.dt, dt = prob.dt/10)
        # get output
        if sol.retcode == ReturnCode.Success
            return Array(sol)'
        else
            println("Error: ", sol.retcode)
            return Array(sol)'
        end
    end

    function calc_AIC(prob::Modified_SINDy_Problem, Ξ::AbstractMatrix)
        # make derivative predictions
        du_pred = prob.Lib(prob.u, prob.alg.x) * Ξ

        residuals = prob.du - du_pred
        rss = sum(residuals.^2)
        n = size(prob.du, 1)
        sigma_squared = rss / n
        likelihood = exp(-0.5 * n * log(2 * π * sigma_squared) - 0.5 * rss / sigma_squared)
        k = count(x-> abs.(x)>0, Ξ)
        return 2 * k - 2 * log(likelihood)
    end

    function calc_RMSE(prob::Modified_SINDy_Problem, Ξ::AbstractMatrix)
        # make derivative predictions
        du_pred = prob.Lib(prob.u, prob.alg.x) * Ξ

        residuals = prob.du - du_pred
        rss = sum(residuals.^2)
        n = size(prob.du, 1)
        return sqrt(rss / n)
    end

    function sparsify(Θ,du,λs,iter; STRRidge = false, ρ=1, abstol=0.000001, reltol = 0.000001)
        # standard sparsify function 
        # Input: 
        #   Θ: The library matrix with size n x p, where n is the data length, p is the number of nonlinear basis. 
        #   du: Estimated or measured derivative of dynamics.
        #   λ: Thresholding parameter.
        #   iter: Number of regressions you would like to perform.
        # return ues: Estimate past state.
        #---------------------------
        # First get the number of states
        n_state=size(du,2)
        
        # Initialize comparison values
        min_loss = 9999999999
        prev_smallinds = [1]

        # Next, perform regression, get an estimate of the selection matrix Ξes
        Ξes = Θ \ du
        X_prev = Θ * Ξes # our first SINDy prediction

        # flag if lambdas are to be automatically determined
        automatic_λ = λs == []

        # determine rho values for STRRidge or STLSQ
        ρs = STRRidge ? ρ*[1+(i-1)/10 for i=1:iter] : fill(ρ, iter)

        for i=1:iter
            # At each iteration, try all λ values to find the best one

            # if automatic lambda, determine the range of lambdas to test
            if automatic_λ
                λs = generate_lambdas(Ξes)
            end

            for λ in λs

                # Get the index of values whose absolute value is smaller than λ
                smallinds = (abs.(Ξes).<λ)

                # If the effect of λ is the same as the previous one, no need to do calculations again
                if smallinds == prev_smallinds
                    continue
                end

                # Make a temporary Ξes matrix to test out the effect of λ
                temp_Ξes = copy(Ξes)
                
                # Set the parameter value of library term whose absolute value is smaller than λ as zero
                temp_Ξes[smallinds].=0

                 # if any column of temp_Ξes is all zeros, then we have a problem
                 if any(sum(abs.(temp_Ξes), dims=1) .== 0)
                    continue
                end

                # Regress the dynamics to the remaining terms
                if STRRidge
                    for ind=1:n_state
                        biginds = .!smallinds[:,ind]
                        M = transpose(Θ[:,biginds])*Θ[:,biginds]
                        temp_Ξes[biginds,ind] = inv(M + ρs[i]*Diagonal(ones(size(M,1))))*(transpose(Θ[:,biginds])*du[:,ind])
                    end
                else
                    for ind=1:n_state
                        biginds = .!smallinds[:,ind]
                        temp_Ξes[biginds,ind] = Θ[:,biginds]\du[:,ind]
                    end
                end

                # Save the current small indices
                prev_smallinds = smallinds

                # calculate the loss and compare it to our best loss
                loss = SINDy_loss(du, Θ, temp_Ξes, ρ)
                if loss < min_loss
                    Ξes = copy(temp_Ξes)
                    min_loss = loss
                end
            end

            X = Θ * Ξes # make new SINDy prediction

            # If nothing, or very little, changed in one iteration, then we have converged
            if _is_converged(X, X_prev, abstol, reltol)
                break
            end
            
            X_prev = X
        end

        return Ξes, min_loss
    end

    function sparsify(prob::Modified_SINDy_Problem)
        # standard sparsify function that accepts a sindy problem 
        #return sparsify(prob.Θ,prob.du,prob.λs,prob.iter; ρ=prob.ρ, abstol=prob.abstol, reltol=prob.reltol)


        # standard sparsify function 
        # Input: 
        #   Θ: The library matrix with size n x p, where n is the data length, p is the number of nonlinear basis. 
        #   du: Estimated or measured derivative of dynamics.
        #   λ: Thresholding parameter.
        #   iter: Number of regressions you would like to perform.
        # return ues: Estimate past state.
        #---------------------------
        # First get the number of states
        n_state=size(prob.du,2)
        
        # Initialize comparison values
        min_loss = 9999999999
        prev_Ξ = [1]

        # Next, perform regression, get an estimate of the selection matrix Ξes
        Ξes = prob.Θ \ prob.du
        X_prev = prob.Θ * Ξes # our first SINDy prediction
        Ξes = Ξes .* prob.active_Ξ

        λs = prob.λs
        automatic_λ = prob.λs == []

        ρs = prob.STRRidge ? prob.ρ*[1+(i-1)/10 for i=1:prob.iter] : fill(prob.ρ, prob.iter)
        #ρs = fill(prob.ρ, prob.iter)

        for i=1:prob.iter
            # At each iteration, try all λ values to find the best one
            #prob.active_Ξ = (abs.(Ξes) .> prob.lower_bounds) .|| (abs.(Ξes) .< prob.upper_bounds)

            # adjust the lambda values if we are using automatic lambda search
            if automatic_λ
                """
                if all(Ξes .== 0) 
                    break
                end
                λmin = min(abs.(Ξes[Ξes .!= 0])...)
                λmax = max(abs.(Ξes[Ξes .!= 0])...)
                λs = range(λmin/10, λmax, 1000*Int(ceil(log10.(λmax/λmin))))
                """
    
                λs = generate_lambdas(Ξes)
                #println(λs)
            end

            for λ in λs
                # Get the index of values whose absolute value is greater than λ
                temp_active_Ξ = copy(prob.active_Ξ)
                prob.active_Ξ = (abs.(Ξes)) .> λ

                # If the effect of λ is the same as the previous one, no need to do calculations again
                if prob.active_Ξ == temp_active_Ξ
                    #prob.active_Ξ = temp_active_Ξ
                    continue
                end

                # Make a temporary Ξes matrix to test out the effect of λ
                temp_Ξes = copy(Ξes)
                
                # Set the parameter value of library term whose absolute value is smaller than λ as zero
                temp_Ξes[.!prob.active_Ξ].=0

                # if any column of temp_Ξes is all zeros, then we have a problem
                if any(sum(prob.active_Ξ, dims=1) .== 0)
                    prob.active_Ξ = temp_active_Ξ
                    continue
                end
                
                # Regress the dynamics to the remaining terms

                if prob.STRRidge 
                    for ind=1:n_state
                        biginds = prob.active_Ξ[:,ind]
                        M = transpose(prob.Θ[:,biginds])*prob.Θ[:,biginds]
                        #display(M)
                        #display(temp_Ξes)
                        temp_Ξes[biginds,ind] = inv(M + ρs[i]*Diagonal(ones(size(M,1))))*(transpose(prob.Θ[:,biginds])*prob.du[:,ind])
                    end
                else
                    for ind=1:n_state
                        biginds = prob.active_Ξ[:,ind]
                        temp_Ξes[biginds,ind] = prob.Θ[:,biginds]\prob.du[:,ind]
                    end
                end

                # Save the current small indices
                prev_Ξ = prob.active_Ξ

                # calculate the loss and compare it to our best loss
                loss = SINDy_loss(prob.du, prob.Θ, temp_Ξes, prob.η)
                #println("loss = ", loss, " for λ = ", λ, " with current min loss = ", min_loss)
                if loss < min_loss
                    Ξes = copy(temp_Ξes)
                    #println("updating Ξes to ", Ξes)
                    min_loss = loss
                else
                    # If the loss is not improved, then we revert the active_Ξ to the previous one
                    prob.active_Ξ = temp_active_Ξ
                end
            end
            #println("finished iteration ", i, " with min loss = ", min_loss, " and Ξes = ", Ξes)

            X = prob.Θ * Ξes # make new SINDy prediction

            # If nothing, or very little, changed in one iteration, then we have converged
            if _is_converged(X, X_prev, prob.abstol, prob.reltol)
                #println("converged")
                break
            end
            
            X_prev = X
        end
        prob.active_Ξ = (abs.(Ξes)) .> 0
        #println("final Ξes = ", prob.active_Ξ)
        return Ξes, min_loss
    end

   function ensemble_solve_SINDy(prob::Modified_SINDy_Problem, batches::Int, pct_size::AbstractFloat, tol::AbstractFloat, parallel::Bool)
        # runs ensemble sparse regression on the problem to get a more robust solution
        if !(prob.alg isa Default_SINDy)
            solve_SINDy(prob, prob.alg)
        end

        # initialize the set of bagging terms
        ΞB = zeros((size(prob.Θ\prob.du)..., batches))
        # determine batch size
        batch_size = Int(floor(pct_size * size(prob.Θ)[1]))
        # perform sparse regression on each batch
        batch_prob = deepcopy(prob)
        """
        for i in 1:batches
            # randomly select batch_size rows from Θ and du
            batch_indices = sort(sample(1:size(prob.du,1), batch_size, replace=false)) # sort the sampled indices
            ΘB = prob.Θ[batch_indices, :]
            duB = prob.du[batch_indices, :]
            # perform sparse regression on the batch
            ΞB[:,:,i], _ = sparsify(ΘB, duB, prob.λs, prob.iter, STRRidge = prob.STRRidge, ρ=prob.ρ, abstol=prob.abstol, reltol=prob.reltol)
        end
        """
        for i in 1:batches
            # randomly select batch_size rows from Θ and du
            batch_indices = sort(sample(1:size(prob.du,1), batch_size, replace=false)) # sort the sampled indices
            batch_prob.Θ = prob.Θ[batch_indices, :]
            batch_prob.du = prob.du[batch_indices, :]
            batch_prob.active_Ξ = prob.control_Ξ
            # perform sparse regression on the batch
            ΞB[:,:,i], _ = sparsify(batch_prob)
        end

        # determine number of times each term appears in the ensemble
        num_appearances = sum(abs.(ΞB) .> 0, dims=3)[:, :, 1]
        # compute the inclusion probability for each term
        ips = num_appearances / batches
        # Compute the ensembled coefficients 
        Ξes = sum(ΞB, dims=3)[:, :, 1] ./ num_appearances
        # probabilistically-filter Ξs based on tol
        Ξes[ips .< tol] .= 0
        prob.active_Ξ .= ips .> tol

        # averaging only when Ξes exactly matches ips
        best_Ξes = zeros(size(Ξes))
        count = 0
        for i in 1:batches
            if all((abs.(ΞB[:,:,i]) .> 0) .== prob.active_Ξ)
                best_Ξes .= best_Ξes .+ ΞB[:,:,i]
                count += 1
            end
        end
        if count > 0
            best_Ξes = best_Ξes ./ count
        end

        #prob.active_Ξ = Matrix{Bool}([1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0])
        #prob.active_Ξ = Matrix{Bool}([1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1])
        Ξes = lsq_regress(prob, prob.ρ)

       return Ξes, ips, best_Ξes
   end

    function CalDerivative(u,dt)
        # Simplest numerical differentiation algorithm (should probably move it ADO_OLE.jl)
        # Input:
        #  u: Measurement data we wish to approxiamte the derivative. 
        #  It should be of size n x m, where n is the number of measurement, m is the number of states.
        # dt: Time step
        # return du: The approximated derivative. 
        #--------------------------- 
    
        # Define the coeficient for different orders of derivative
        p1=1/12;p2=-2/3;p3=0;p4=2/3;p5=-1/12;
        
        du=(p1*u[1:end-4,:]+p2*u[2:end-3,:]+p3*u[3:end-2,:]+p4*u[4:end-1,:]+p5*u[5:end,:])/dt;
            
        return du
    end
end