import Pkg; Pkg.add("JuMP")
import Pkg; Pkg.add("OSQP")
using JuMP
#using Plots
using LinearAlgebra

#! Old model parameters
# #defining all parameters for model and probem
# A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
#         9.83e-4 1 -4.09e-8 -8.32e-5;
#         4.83e-7 9.83e-4 1 -5.34e-4;
#         1.58e-10 4.83e-7 9.83e-4 .9994;
# ]

# B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'

# B1 = B

# # fake opsin
# Binh = -B .* (1 .+ randn(4, 1)./5)
# B2 = [B Binh]

# C = [-.0096 .0135 .005 -.0095]

y2z(y) = exp(61.4*y - 5.468)
z2y(z) = (log(z) + 5.468)/61.4

t_step = 0.001 #1 milisecond
pred = 30 #prediction horizon
ctr = 6 #control horizon
sample = 200 #number of steps between sampling points for control

R = 1e-6*I # original R = 1e-5*I
T = 13000
Tpred = pred*sample

using OSQP

#function now designed not to do a control loop but only solve the optimization problem once
#assumes that reference has been "shifted" properly
#renamed to beta since new mpc function below takes variable parameters and is thus more flexible
function mpc_beta(x0, zref; nu=1, u_clamp=nothing, sample=250)
    if nu == 1
        B = B1
    elseif nu == 2
        B = B2
    end

    Tpred = pred*sample

    if length(zref) < Tpred
        zrefpad = cat(zref, fill(zref[end], Tpred - length(zref)), dims=1)
    else    
        zrefpad = zref
    end
    
    #println(typeof(zrefpad))
    neuron = Model(OSQP.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr+1)] .<= 40
        yD[i=1:1, t=1:Tpred]
        # xD[i=1:4, t=1:Tpred], (start = 0)
    end)

    @expressions(
        neuron,
        begin
            y, C*x
            # x_error[t=1:Tpred], x[:, t] - xD[:, t]
            # x_cost[t=1:Tpred], x_error[t]'*Q*x_error[t]
            y_error[t=1:Tpred], y[:, t] - yD[:, t]
            y_cost[t=1:Tpred], y_error'[t]*y_error[t]
            # sampled_x_cost[t=1:pred], x_cost[t*sample]
            sampled_y_cost[t=1:pred], y_cost[t*sample]
            u_cost[t=1:ctr+1], u[t]'*R*u[t]
        end
    )

    #fix first sample steps
    @constraint(neuron, x[:, 2:(sample)] .== A*x[:, 1:sample-1] + B*u[:, 1])

    #fix each sample period
    for i in 1:(ctr-1)
        @constraint(neuron, x[:, (sample*i+1):(sample*i+sample)] .== A*x[:, (sample*i):(sample*i+sample-1)] + B*u[:, (i+1)])
    end

    #fix rest of inputs 
    @constraint(neuron, x[:, (ctr*sample+1):(Tpred)] .== A*x[:, (ctr*sample):(Tpred-1)] + B*u[:, (ctr+1)])
    yDall = z2y.(zrefpad)
    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    # J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    J = @objective(neuron, Min, sum(sampled_y_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    # if nu == 2 
    #     B = [B -B]
    # end

    x_current = x0

    #for t in 1:steps - got rid of for loop since only optimizing once
    fix.(x[:, 1], x_current; force=true)
    if u_clamp != nothing
        fix.(u[:, 1], u_clamp; force=true)
    end

    #set desired trajectory and optimize
    fix.(yD[:], yDall[1:Tpred], force=true)
    optimize!(neuron)

    #return first input for use in experiment
    optimal_u = value.(u[:, 1])
    return optimal_u
end

###new version which will take in variable parameters
function mpc_v2(x0, zref; nu=1, u_clamp=nothing, sample=250, A=A, B=B, C=C)
    # if nu == 1
    #     B = B1
    # elseif nu == 2
    #     B = B2
    # end

    Tpred = pred*sample

    if length(zref) < Tpred
        zrefpad = cat(zref, fill(zref[end], Tpred - length(zref)), dims=1)
    else    
        zrefpad = zref
    end
    #println(typeof(zrefpad))
    neuron = Model(OSQP.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr+1)] .<= 40
        yD[i=1:1, t=1:Tpred]
        # xD[i=1:4, t=1:Tpred], (start = 0)
    end)

    @expressions(
        neuron,
        begin
            y, C*x
            # x_error[t=1:Tpred], x[:, t] - xD[:, t]
            # x_cost[t=1:Tpred], x_error[t]'*Q*x_error[t]
            y_error[t=1:Tpred], y[:, t] - yD[:, t]
            y_cost[t=1:Tpred], y_error'[t]*y_error[t]
            # sampled_x_cost[t=1:pred], x_cost[t*sample]
            sampled_y_cost[t=1:pred], y_cost[t*sample]
            u_cost[t=1:ctr+1], u[t]'*R*u[t]
        end
    )

    #fix first sample steps
    @constraint(neuron, x[:, 2:(sample)] .== A*x[:, 1:sample-1] + B*u[:, 1])

    #fix each sample period
    for i in 1:(ctr-1)
        @constraint(neuron, x[:, (sample*i+1):(sample*i+sample)] .== A*x[:, (sample*i):(sample*i+sample-1)] + B*u[:, (i+1)])
    end

    #fix rest of inputs 
    @constraint(neuron, x[:, (ctr*sample+1):(Tpred)] .== A*x[:, (ctr*sample):(Tpred-1)] + B*u[:, (ctr+1)])
    yDall = z2y.(zrefpad)
    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    # J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    J = @objective(neuron, Min, sum(sampled_y_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    # if nu == 2 
    #     B = [B -B]
    # end

    x_current = x0

    #for t in 1:steps - got rid of for loop since only optimizing once
    fix.(x[:, 1], x_current; force=true)
    if u_clamp != nothing
        fix.(u[:, 1], u_clamp; force=true)
    end

    #set desired trajectory and optimize
    fix.(yD[:], yDall[1:Tpred], force=true)
    optimize!(neuron)

    #return first input for use in experiment
    optimal_u = value.(u[:, 1])
    return optimal_u
end

#!Current version of mpc
#Called flex_mpc since can take variable parameters and switch between firing rate and lfp input
#1 stands for z mode and 2 is for y mode - did this because waned to avoid converting julia and python strings
function flex_mpc(x0, ref; nu=1, sample=250, A=A, B=B, C=C, ref_type=1)
    # if nu == 1
    #     B = B1
    # elseif nu == 2
    #     B = B2
    # end

    Tpred = pred*sample
    # See if we were given a y or z reference
    if ref_type == 1
        if length(ref) < Tpred
            zrefpad = cat(ref, fill(ref[end], Tpred - length(ref)), dims=1)
        else    
            zrefpad = ref
        end
        yDall = z2y.(zrefpad)
    elseif ref_type == 2
        if length(ref) < Tpred
            yrefpad = cat(ref, fill(ref[end], Tpred - length(ref)), dims=1)
        else    
            yrefpad = ref
        end
        yDall = yrefpad
    end

    #println(typeof(zrefpad))
    neuron = Model(OSQP.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr+1)] .<= 70
        yD[i=1:1, t=1:Tpred]
        # xD[i=1:4, t=1:Tpred], (start = 0)
    end)

    @expressions(
        neuron,
        begin
            y, C*x
            # x_error[t=1:Tpred], x[:, t] - xD[:, t]
            # x_cost[t=1:Tpred], x_error[t]'*Q*x_error[t]
            y_error[t=1:Tpred], y[:, t] - yD[:, t]
            y_cost[t=1:Tpred], y_error'[t]*y_error[t]
            # sampled_x_cost[t=1:pred], x_cost[t*sample]
            sampled_y_cost[t=1:pred], y_cost[t*sample]
            u_cost[t=1:ctr+1], u[t]'*R*u[t]
        end
    )

    #fix first sample steps
    @constraint(neuron, x[:, 2:(sample)] .== A*x[:, 1:sample-1] + B*u[:, 1])

    #fix each sample period
    for i in 1:(ctr-1)
        @constraint(neuron, x[:, (sample*i+1):(sample*i+sample)] .== A*x[:, (sample*i):(sample*i+sample-1)] + B*u[:, (i+1)])
    end

    #fix rest of inputs 
    @constraint(neuron, x[:, (ctr*sample+1):(Tpred)] .== A*x[:, (ctr*sample):(Tpred-1)] + B*u[:, (ctr+1)])
    # yDall = z2y.(zrefpad)
    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    # J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    J = @objective(neuron, Min, sum(sampled_y_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    # if nu == 2 
    #     B = [B -B]
    # end

    x_current = x0

    #for t in 1:steps - got rid of for loop since only optimizing once
    fix.(x[:, 1], x_current; force=true)

    #set desired trajectory and optimize
    fix.(yD[:], yDall[1:Tpred], force=true)
    optimize!(neuron)

    #return first input for use in experiment
    optimal_u = value.(u[:, 1])
    return optimal_u
end

# ? Open loop mpc function
function open_loop_mpc(x0, ref; nu=1, sample=3, A=A, B=B, C=C, ref_type=1)
    # if nu == 1
    #     B = B1
    # elseif nu == 2
    #     B = B2
    # end
    print("\nref:", ref,"\n")
    Tpred = length(ref)
    print("\ngot length\n")
    pred = length(ref)
    ctr = Int(trunc(Int, length(ref)/sample)) #must be integer for indexing later
    
    # See if we were given a y or z reference
    if ref_type == 1
        zrefpad = ref
        yDall = z2y.(zrefpad)
    elseif ref_type == 2 
        yDall = ref
    end

    #println(typeof(zrefpad))
    neuron = Model(OSQP.Optimizer)
    set_silent(neuron)
    print("\ngot model\n")

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr)] .<= 70 #was ctr+1 ??? why
        yD[i=1:1, t=1:Tpred]
        # xD[i=1:4, t=1:Tpred], (start = 0)
    end)
    print("\ngot vars\n")

    @expressions(
        neuron,
        begin
            y, C*x
            # x_error[t=1:Tpred], x[:, t] - xD[:, t]
            # x_cost[t=1:Tpred], x_error[t]'*Q*x_error[t]
            y_error[t=1:Tpred], y[:, t] - yD[:, t]
            y_cost[t=1:Tpred], y_error'[t]*y_error[t]
            # sampled_x_cost[t=1:pred], x_cost[t*sample]
            #sampled_y_cost[t=1:pred], y_cost[t*sample]
            u_cost[t=1:ctr], u[t]'*R*u[t] #was ctr + 1
        end
    )
    print("\ngot expressions\n")

    #fix first sample steps
    @constraint(neuron, x[:, 2:(sample)] .== A*x[:, 1:sample-1] + B*u[:, 1])

    #fix each sample period
    for i in 1:(ctr-1)
        @constraint(neuron, x[:, (sample*i+1):(sample*i+sample)] .== A*x[:, (sample*i):(sample*i+sample-1)] + B*u[:, (i+1)])
    end

    #fix rest of inputs 
    @constraint(neuron, x[:, (ctr*sample+1):(Tpred)] .== A*x[:, (ctr*sample):(Tpred-1)] + B*u[:, (ctr)]) #was ctr+1
    # yDall = z2y.(zrefpad)
    # x_cost[t] returns a 1x1 matrix, which we need to index to get the value out
    # J = @objective(neuron, Min, sum(sampled_x_cost[t] for t in 2:(pred)) + sum(u_cost[t] for t in 1:ctr+1))
    J = @objective(neuron, Min, sum(y_cost[t] for t in 2:(Tpred)) + sum(u_cost[t] for t in 1:ctr))
    # if nu == 2 
    #     B = [B -B]
    # end

    x_current = x0

    #for t in 1:steps - got rid of for loop since only optimizing once
    fix.(x[:, 1], x_current; force=true)

    #set desired trajectory and optimize
    fix.(yD[:], yDall[1:Tpred], force=true)
    optimize!(neuron)

    #return first input for use in experiment
    optimal_us_vec = value.(u[:, 1:ctr])
    return optimal_us_vec
end