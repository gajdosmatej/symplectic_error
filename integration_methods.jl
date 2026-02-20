using ProgressLogging
include("./physics_calculations.jl")


"""
Step the current position and momentum state using the p-implicit Symplectic Euler method 
(which is in this case just a linear system).
"""
function stepQuasiexplicit!(position, momentum, h, constants)
	A_point, DA_T_point = getAAndDA_T(position, constants)

	momentum .= (I - h*DA_T_point) \ (momentum - h*DA_T_point*A_point)
	position .= position + h*getHp(position, momentum, constants)
end


"""
Step the current position and momentum state using the q-implicit Symplectic Euler method. 
Use num_FPI fixed point iterations.
"""
function stepImplicit!(position, momentum, h, constants, num_FPI)
	temp_pos = position
	for i=1:num_FPI
		temp_pos = position + h*getHp(temp_pos, momentum, constants)
	end
	position .= temp_pos
	momentum .= momentum - h*getHq(position, momentum, constants)
end


"""
Integrate the Hamiltonian system with given initial conditions with either p-implicit or q-implicit Symplectic Euler method,
based on the use_implicit value. Return arrays (xs[num_plotpoints], ys[num_plotpoints], zs[num_plotpoints]) 
of the integrated positions (on a coarser mesh defined by the num_plotpoints argument).
"""
function integrate(constants, initial_conditions, h, num_iter, num_plotpoints=1000, use_implicit=false, FPI_iters=1)
	modus = div(num_iter, num_plotpoints)
	
	step_method! = stepQuasiexplicit!

	if use_implicit
		step_method! = (position_, momentum_, h_, constants_) -> stepImplicit!(position_, momentum_, h_, constants_, FPI_iters)	# currying stepImplicit(., FPI_iters)
	end

	xs_plot, ys_plot, zs_plot = zeros(num_plotpoints), zeros(num_plotpoints), zeros(num_plotpoints)
	position = 1/constants.L_DIM * initial_conditions.q0
	momentum = 1/constants.P_DIM * initial_conditions.p0
	@progress for i=1:num_iter
		if mod(i-1, modus) == 0
			k = div(i-1, modus) + 1
			xs_plot[k], ys_plot[k], zs_plot[k] = constants.L_DIM * position
		end

		step_method!(position, momentum, h, constants)
	end
	return xs_plot, ys_plot, zs_plot
end


"""
Integrate the Hamiltonian system with given initial conditions with either p-implicit or q-implicit Symplectic Euler method,
based on the use_implicit value. Return array Hs[num_plotpoints] 
of the Hamiltonian values along integration (on a coarser mesh defined by the num_plotpoints argument).
"""
function integrateReturnHamiltonian(constants, initial_conditions, h, num_iter, num_plotpoints=1000, use_implicit=false, FPI_iters=1)
	modus = div(num_iter, num_plotpoints)
	
	step_method! = stepQuasiexplicit!

	if use_implicit
		step_method! = (position_, momentum_, h_, constants_) -> stepImplicit!(position_, momentum_, h_, constants_, FPI_iters)	# currying stepImplicit(., FPI_iters)
	end

	Hs_plot = zeros(num_plotpoints)
	position = 1/constants.L_DIM * initial_conditions.q0
	momentum = 1/constants.P_DIM * initial_conditions.p0
	@progress for i=1:num_iter
		if mod(i-1, modus) == 0
			k = div(i-1, modus) + 1
			Hs_plot[k] = constants.H_DIM * getH(position, momentum, constants)
		end

		step_method!(position, momentum, h, constants)
	end
	return Hs_plot
end


"""
Return new dimensionless position and momenta values using the q-implicit Symplectic Euler method with FPI_iters fixed point iterations.
"""
function implicitFlow(position, momentum, constants, h, FPI_iters)
	position_new = position
	for i=1:FPI_iters
		position_new = position + h*getHp(position_new, momentum, constants)
	end
	momentum_new = momentum - h*getHq(position_new, momentum, constants)
	return [position_new; momentum_new]
end


"""
Return the Jacobi matrix of the flow given by one step of the q-implicit Symplectic Euler method 
with FPI_iters fixed point iterations.
"""
function getFlowDifferential(position, momentum, constants, h, FPI_iters)
	flow_as_state_func = ( state_ -> implicitFlow(state_[1:3], state_[4:6], constants, h, FPI_iters) )
	return ForwardDiff.jacobian(flow_as_state_func, [position; momentum])
end


"""
Get the perturbation of the symplectic structure matrix J after applying one step of the q-implicit Symplectic Euler method
with FPI_iters fixed point iterations.
"""
function getPerturbedMatrix(position, momentum, constants, h, FPI_iters)
	J = [	0 0 0 1 0 0; 
				0 0 0 0 1 0;
				0 0 0 0 0 1;
				-1 0 0 0 0 0;
				0 -1 0 0 0 0;
				0 0 -1 0 0 0
			]
	DPhi = getFlowDifferential(position, momentum, constants, h, FPI_iters)
	return transpose(DPhi) * J * DPhi
end