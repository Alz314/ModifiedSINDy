module smoothing
    using Plots
    using LinearAlgebra
    using Distributions
    using Optimization, OptimizationOptimJL
    using DataDrivenDiffEq
    using SavitzkyGolay
    using StatsBase

    export SG_smoothing, SG_smoothing_optim, smooth_with_kernel

    function SG_smoothing(un, window_size, dt; order = 6)
        """
        Performs Savitzky Golay smoothing on noisy data and returns smoothed data as well as derivative estimates.
        Window size must be manually specified
    
        # Arguments
        * `un`: AbstractArray type holding the noisy data to be smoothed
        * `window_size`: Int type holding the window size to be used
        * `dt`: Float type holding the timestep spacing between points; defaults to 6 which should be fine
        * `order`: Int type that holds the order of polynomial fitting to be used
        """
        ues, dues = SG_smoothing_1D(un[:, 1], dt, window_size, order)
        for i in 2:size(un)[2]
            ues_next, dues_next = SG_smoothing_1D(un[:, i], dt, window_size, order)
            ues = hcat(ues, ues_next)
            dues = hcat(dues, dues_next)
        end
        return ues, dues
    end

    function SG_smoothing_optim(un, dt; order = 6, opt_alg = ParticleSwarm(), print_window_size = false, disp_loss_landscape = false, loss_function = 1)
        """
        Performs Savitzky Golay smoothing on noisy data and returns smoothed data as well as derivative estimates.
        Everything is automatically optimized, but you can change the order if you want

        # Arguments
        * `un`: AbstractArray type holding the noisy data to be smoothed
        * `dt`: Float type holding the timestep spacing between points; defaults to 6 which should be fine
        * `order`: Int type that holds the order of polynomial fitting to be used
        * `opt_alg`: Optimization algorithm; NelderMead is okay, but ParticleSwarm is probably better
        * `disp_loss_landscape`: boolean to plot the entire loss landscape of different window values (can take a while)
        * `print_window_size`: boolean to print the exact window size that was used
        * `loss_function`: integer to choose which loss function to use; 1 is autocor + crosscor, 2 is weighted autocor
        """
        ws = optim_window_size(un[:, 1], dt; order = order, opt_alg = opt_alg, disp_loss_landscape = disp_loss_landscape, loss_function = loss_function)
        print_window_size && print(ws)
        ues, dues = SG_smoothing_1D(un[:, 1], dt, ws, order)
        for i in 2:size(un)[2]
            ues_next, dues_next = SG_smoothing_1D(un[:, i], dt, ws, order)
            ues = hcat(ues, ues_next)
            dues = hcat(dues, dues_next)
        end
        return ues, dues
    end

    function smooth_with_kernel(un, T, kernel)
        """
        Performs smoothing on noisy data using a kernel function and returns smoothed data as well as derivative estimates.
        
        # Arguments
        * `un`: AbstractArray type holding the noisy data to be smoothed
        * `T`: AbstractArray type holding the time data
        * `kernel`: kernel from the DataDrivenDiffEq package
        """

        if (size(un)[1] == size(T)[1])
            dues, ues = DataDrivenDiffEq.collocate_data(transpose(un), T, kernel)
        elseif (size(un[1:end-1, :])[1] == size(T)[1])
            dues, ues = DataDrivenDiffEq.collocate_data(transpose(un[1:end-1, :]), T, kernel)
        elseif (size(un)[2] == size(T)[1])
            dues, ues = DataDrivenDiffEq.collocate_data(un, T, kernel)
        else
            datashape = size(un)
            timeshape = size(T)
            error("Dimensions of inputs do not match: Data = $datashape and time = $timeshape")
        end
        return transpose(ues), transpose(dues)
    end

    function SG_smoothing_1D(un, dt, window, order)
        # Performs Savitzky Golay smoothing on just one dimension of data
        # This is a helper function, you shouldn't need to call it

        ues = savitzky_golay(un, window, order).y
        dues = savitzky_golay(un, window, order, deriv=1, rate=1/dt).y
        return ues, dues
    end

    function optim_window_size(un, dt; order = 5, opt_alg = ParticleSwarm(), disp_loss_landscape = false, loss_function = 1)
        # Optimizes window size for Savitzky Golay smoothing

        function loss(w, p)
            # input validation of window size
            wn = w[1]
            if isnan(wn) || wn < order+3 || wn > size(un)[1]
                return 10000000 # returns very high loss if window size is beyond bounds
            end
            wn = iseven(round(wn)) ? Int(round(wn) - 1) : Int(round(wn))

            noise = un .- savitzky_golay(un, wn, order).y # estimated noise from savitzky golay smoothing

            if loss_function == 1
                # first term checks if any lags of the noise repeat, which would imply that the smoothing missed features of the actual function
                # second term checks if the noise is correlated to our initial data, which it shouldn't (smoothing the noise estimate improves performance)
                return mean(abs.(autocor(noise))[2:end]) + abs(crosscor(un, savitzky_golay(noise, wn, order).y, [0])[1]) 
            elseif loss_function == 2
                a = autocor(noise)[2:end]
                weights = exp.((0:length(a)-1)/length(a)) # could experiment with this weighting function
                return sum(abs, a.*weights) # weights penalize longer time correlations more severely
            elseif loss_function == 3
                du_noisy = savitzky_golay(un, wn, order, deriv=1, rate=1/dt).y
                dues = savitzky_golay(du_noisy, wn, order).y
                noise_du = du_noisy .- dues
                return mean(abs.(autocor(noise_du))[2:end])
            end
        end

        if disp_loss_landscape
            xs = collect((order*2 + 1):2:size(un)[1])
            #xs = collect((order*2 + 1):2:100)
            ys = zeros(size(xs))
            for (i, w) in enumerate(xs)
                ys[i] = loss([w], [])
            end
            display(plot(xs, ys))
        end

        #initial guess
        w = 2.0 * order + 1

        optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
        optprob = OptimizationProblem(optf, [w]) 
        opt_w = solve(optprob, opt_alg).u[1]
        opt_w = iseven(round(opt_w)) ? Int(round(opt_w) - 1) : Int(round(opt_w))
        return opt_w
    end
end