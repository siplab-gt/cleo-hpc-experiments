using JuMP
using Plots
using LinearAlgebra

#defining all parameters for model and probem
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
        9.83e-4 1 -4.09e-8 -8.32e-5;
        4.83e-7 9.83e-4 1 -5.34e-4;
        1.58e-10 4.83e-7 9.83e-4 .9994;
]

B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'

B1 = B

# fake opsin
Binh = -B .* (1 .+ randn(4, 1)./5)
B2 = [B Binh]

C = [-.0096 .0135 .005 -.0095]

y2z(y) = exp(61.4*y - 5.468)
z2y(z) = (log(z) + 5.468)/61.4

t_step = 0.001 #1 milisecond
pred = 25 #prediction horizon
ctr = 5 #control horizon
sample = 200 #number of steps between sampling points for control

zD = 0.2
yD = z2y(zD)
uD = inv(C*inv(I - A)*B) * yD
#xD = inv(I - A) * B * uD
Q = C'*C
R = 1e-5*I
T = 13000
Tpred = pred*sample

# add in our variable reference
#structured to take a list of firing rate values, convert them to x vectors, then pad with zeros if necessary
##zref = [i < 3000 ? 0.1 : 0.15 for i in 1:T] #trial firing rate reference
# trial firing rate reference
zref = .1 .+ .08*sin.(range(start=0, stop=2*pi, length=Int(1.5*Tpred)));

using OSQP

#function now designed not to do a control loop but only solve the optimization problem once
function mpc(steps, x0, zref; nu=1, u_clamp=nothing, sample=250)
    if nu == 1
        B = B1
    elseif nu == 2
        B = B2
    end

    Tpred = pred*sample
    Tall = steps*sample

    if length(zref) < Tall + Tpred
        zrefpad = cat(zref, fill(zref[end], Tall + Tpred - length(zref)), dims=1)
    end

    neuron = Model(OSQP.Optimizer)
    set_silent(neuron)

    #Define state variables
    @variables(neuron, begin
        x[i=1:4, t=1:Tpred]
        0 ≤ u[1:nu, 1:(ctr+1)]
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

    #now need to update the reference by "shifting it" one sample size forward and padding with end value
    fix.(yD[:], yDall[1:Tpred], force=true)
    optimize!(neuron)

    #return first input for use in experiment
    optimal_u = value.(u[:, 1])
    return optimal_u

    #     zs[1, tfine] = y2z.(value(y[1]))
    #     us[:, tfine] = value.(u[:, 1])
    #     # append!(zs , y2z.(value(y[1])))
    #     # append!(us, value(u[:, 1]))
    #     x_current = value.(x[:, 2])
    #     # println(x_current)
    #     # now effectively apply optimal first u for 250 more steps
    #     const_u = value.(u[:, 1]);
    #     tfine += 1

    #     for i in 1:(sample-1)
    #         x_current = A*x_current + B*const_u
    #         # append!(zs, y2z.((C*x_current)[1]))
    #         zs[1, tfine] = y2z.((C*x_current)[1])
    #         us[:, tfine] = const_u
    #         tfine += 1
    #     end
    # end
    # #println(solution_summary(neuron))
    # return zs, us
end

steps = T ÷ sample
mpc2res = mpc(steps, zeros(4), zref, nu=2, u_clamp=nothing, sample=sample)