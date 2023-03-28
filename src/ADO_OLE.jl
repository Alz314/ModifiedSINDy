module ADO_OLE_Module
    export solve_SINDy
    export ADO

    import ..SINDy_Base: SINDy_Alg, Modified_SINDy_Problem, sparsify, solve_SINDy, CalDerivative
    using DifferentialEquations, ForwardDiff
    using Flux
    using Distributions


    # New Data types
    # ----------------------------------------------------------------------------------
    mutable struct ADO <: SINDy_Alg
        x::Vector{Float64}
        q::Int16
        ω::Float64
        Nloop::Int16
        Ntrain::Int16
        opt::Flux.Optimise.AbstractOptimiser
        verbose::Bool
    end


    # Constructors for convenience
    # ----------------------------------------------------------------------------------


    
    # Functions 
    # ----------------------------------------------------------------------------------
    function ODE_RHS(u,x,Ξ)
        # u: The measuremt state matrix with size n x m, where n is the data length, m is the number of states. 
        # Ξ: Selection matrix with size p x m, where p is the number of candidate terms in the library, m is the number of state variables. 
        #--------------------------- 
        du=Lib(u,x)*Ξ
         
        return du
         
     end
    
     function RK_45_Forward(u,x,Ξ,dt)
        # u: The measuremt state matrix with size n x m, where n is the data length, m is the number of states. 
        # Ξ: Selection matrix with size p x m, where p is the number of candidate terms in the library, m is the number of state variables.
        # dt: Time step.
        # return ues: Estimate future state.
        #---------------------------
        Kf1=ODE_RHS(u,x,Ξ)*dt
    
        dum1=u+0.5*Kf1
        Kf2=ODE_RHS(dum1,x,Ξ)*dt
    
        dum2=u+0.5*Kf2
        Kf3=ODE_RHS(dum2,x,Ξ)*dt
    
        dum3=u+Kf3
        Kf4=ODE_RHS(dum3,x,Ξ)*dt
        #
        ues=u.+(1/6)*(Kf1.+2*Kf2.+2*Kf3.+Kf4)
        
        return ues
    end
    
    function RK_45_Backward(u,x,Ξ,dt)
        # u: The measuremt state matrix with size n x m, where n is the data length, m is the number of states. 
        # Ξ: Selection matrix with size p x m, where p is the number of candidate terms in the library, m is the number of state variables.
        # q: backward step.
        # dt: Time step.
        # return ues: Estimate past state.
        #---------------------------
        Kb1=ODE_RHS(u,x,Ξ)*dt
    
        dum1=u-0.5*Kb1
        Kb2=ODE_RHS(dum1,x,Ξ)*dt
    
        dum2=u-0.5*Kb2
        Kb3=ODE_RHS(dum2,x,Ξ)*dt
    
        dum3=u-Kb3
        Kb4=ODE_RHS(dum3,x,Ξ)*dt
        #
        ues=u.-(1/6)*(Kb1.+2*Kb2.+2*Kb3.+Kb4)
    
        return ues
    end
    
    function ADO_loss(Measurement,NoiseEs,x,Ξes,Ξactive,q,dt,ω)
        # Measurement: Measurment matrix with size of n x m, where n is the data length and m is the number state variables.
        # NoiseEs: Estimate of noise added to signal.
        # Ξes: Estimated selection vector.
        # x: Extended variable estimates
        # Ξactive: Activation matrix. If the elements in Ξactive is zero, then it means the correpsonding basis in the library is set to inactive.
        # q: Forward/backward simulation step.
        # dt: Time step.
        # ω: Decaying factor of loss value.
        # return LossVal: Value of loss function.
        #--------------------------- 
        # The selection vector that defines our ODE's right hand side should exclude the inactive nonlinear terms
        Ξdummy=Ξes.*Ξactive
        
        # Define matrix to store value
        FutureEstimate=[]
        PastEstimate=[]
        lossFuture=0
        lossPast=0
        lossEstimate=0
        
        # Simulate system based on intial conditions
        for i=1:q
            # If this is the first prediction
            if i==1
                # Estimate the future state, the initial condition's index ranges from t=1 to t=n-q, thus the estimated states'
                # index will range from t=2 to t=n-q+1
                FutureEstimate=RK_45_Forward(Measurement[1:end-q,:]-NoiseEs[1:end-q,:],x,Ξdummy,dt)
                
                # Estimate the past state, the initial condition's index ranges from t=q+1 to t=n, thus the estimated states'
                # index will range from t=q to t=n-1
                PastEstimate=RK_45_Backward(Measurement[q+1:end,:]-NoiseEs[q+1:end,:],x,Ξdummy,dt)
                
                # Calculate the future estimate loss
                lossFuture=norm(Measurement[1+i:end-q+i,:]-NoiseEs[1+i:end-q+i,:]-FutureEstimate).^2
                
                # Calculate the past estimate loss
                lossPast=norm(Measurement[q+1-i:end-i,:]-NoiseEs[q+1-i:end-i,:]-PastEstimate).^2
                
                # Add future and past loss together
                lossEstimate=(ω^(i-1))*(lossFuture+lossPast)
            else
                # Again estimate the future state one step forward based on previous simulated state value
                # The estimated states' index will range from t=1+i to t=n-q+i this time.
                FutureEstimate=RK_45_Forward(FutureEstimate,x,Ξdummy,dt) 
    
                # Similarly, estimate the past state one step backward based on previous simulated state
                # The estimated states' index will range from t=q+1-i to t=n-i this time.
                PastEstimate=RK_45_Backward(PastEstimate,x,Ξdummy,dt) 
    
                # Calculate the future estimate loss
                lossFuture=norm(Measurement[1+i:end-q+i,:]-NoiseEs[1+i:end-q+i,:]-FutureEstimate).^2
                
                # Calculate the past estimate loss
                lossPast=norm(Measurement[q+1-i:end-i,:]-NoiseEs[q+1-i:end-i,:]-PastEstimate).^2
                
                # Add future and past loss together
                lossEstimate=lossEstimate+(ω^(i-1))*(lossFuture+lossPast)
            end
        end
                
        # After calculated the simulation error, calculate the derivative error
        #Calculate the numerical derivative on the estimated true state to get the approximation of true derivative
        EstimatedDerivative_LHS=CalDerivative(Measurement-NoiseEs,dt)
        
        #Calculate the derivative estimate by ODE right hand side
        EstimatedDerivative_RHS=Lib(Measurement[3:end-2,:]-NoiseEs[3:end-2,:],x)*Ξdummy
        
        #Calculate the loss of two derivative estimates
        lossDerivative=norm(EstimatedDerivative_LHS-EstimatedDerivative_RHS).^2
        
        # Finally, add all the loss together
        lossVal=lossEstimate+lossDerivative
        
        return lossVal
    end
    
    function trainLoss(Measurement,NoiseEs,x,Ξes,Ξactive,q,dt,ω,Ntrain,θ,opt, verbose)
        # Measurement: Measurment matrix with size of n x m, where n is the data length and m is the number state variables.
        # NoiseEs: Estimate of noise added to signal.
        # Ξes: Estimated selection vector.
        # Ξactive: Activation matrix. If the elements in Ξactive is zero, then it means the correpsonding basis in the library is set to inactive.
        # q: Forward/backward simulation step.
        # dt: Time step.
        # ω: Decaying factor of loss value.
        # Ntrain: Iteration of the optimizer.
        # θ: Optimization variable
        # opt: Optimizer.
        # return: none
        #--------------------------- 
        for i=1:Ntrain
            # Get the gradient of loss function
            grads=gradient(()->ADO_loss(Measurement,NoiseEs,x,Ξes,Ξactive,q,dt,ω),θ)
            
            # Update the optimization variable
            for p in θ
                Flux.Optimise.update!(opt, p, grads[p])
            end
            
            # Print training process
            if verbose && mod(i,Int(floor(Ntrain/5)))==0
                println("\t $((i/(Ntrain/5))*20) percent finished...Current cost value is $(ADO_loss(Measurement,NoiseEs,x,Ξes,Ξactive,q,dt,ω))")
            end
        end
            
    end
    
    function solve_SINDy(prob::Modified_SINDy_Problem, ADO_params::ADO)
        # Takes in Modified_SINDy_Problem type and solves using a combination of ADO and OLE
        #---------------------------
        x = ADO_params.x
    
        # First estimate the noise, here we do not use any pre-smoothing and set the initial estimate of noise as zeros
        NoiseEs=zeros(size(prob.du))
        
        # After estimating the noise, we can estimate the true state by subtracting estimated noise from measurement
        StateEs0=prob.u-NoiseEs
        
        # Get the number of states
        n_state=size(prob.du,2)
        
        # Based on the estimated state, we can estimate the derivative of the state
        StateDerivativeEs=CalDerivative(StateEs0,prob.dt)
    
        # Once we get the estimated state and its corresponding derivative, we can estimate the value of Ξes
        prob.Θ=prob.Lib(StateEs0[3:end-2,:], x)
        Ξes, loss_SINDy = sparsify(prob)
        X_prev = prob.Lib(StateEs0[4:end-3,:], x) * Ξes
        
        # When first started, set the activation matrix as ones
        Ξactive=ones(size(Ξes))
    
    
        # Now start training loop
        for j=1:ADO_params.Nloop
            # Define optimization variables we would like to train
            θ=Flux.params([NoiseEs,x,Ξes])
            
            ADO_params.verbose && println("Loop $(j)...")
            
            # Training statrt!
            trainLoss(prob.du, NoiseEs, x, Ξes, Ξactive, ADO_params.q, prob.dt, ADO_params.ω, ADO_params.Ntrain, θ, ADO_params.opt, ADO_params.verbose)
            
            ADO_params.verbose && println("Loop $(j) finished!")
    
            # Get the new estimate of state, and transfer this to cpu
            # Note: Here we discarded first and last q points since those point are only 
            # constrained using forward or backward dynamics, and not the both.
            StateEs=prob.du[ADO_params.q+1:end-ADO_params.q-1,:]-NoiseEs[ADO_params.q+1:end-ADO_params.q-1,:]
            
            # Bsed on the updated state estimate, calculate the estimated derivative
            StateDerivativeEs=CalDerivative(StateEs,prob.dt)
            
            # Get the new library matrix based on updated state estimate 
            prob.Θ=prob.Lib(StateEs[3:end-2,:], x)
    
            Ξes, _ = sparsify(prob)
            
    
            # Display the new parameter matrix
            if ADO_params.verbose
                println("Current value of the Ξ is shown below...\n")
                display(Ξes)
                println("Current value of x is shown below...\n")
                display(x)
            end
        end
        ADO_params.x = x
        return NoiseEs, Ξes, x
    end
end