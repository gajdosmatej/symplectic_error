using LinearAlgebra
using ForwardDiff
include("./structs.jl")


function getRho(x_dim, y_dim)
	return sqrt(x_dim^2 + y_dim^2)
end


function getR(rho, z_dim, constants)
	return sqrt(z_dim^2 + (rho - constants.R)^2)
end


"""
Calculate primitive function of f = r/(R*q).
"""
function getF(r, constants)
	a, R = constants.a, constants.R
	return 1/R * (1/(3a) * r^3 − 1/(2a^2) * r^2 + (1+a^2)/a^3 * r − (1 + a^2)/a^4 * log(a*r + 1))
end


"""
Calculate the dimensionless vector potential from the rho, x, y and F values. Auxiliary function.
"""
function getA(rho, F, x_dim, y_dim, constants)
	B0, R = constants.B0, constants.R
	Ax_dim = B0*F/rho * (-y_dim/rho)
	Ay_dim = B0*F/rho * (x_dim/rho)
	Az_dim = -B0*R*log(rho/R)
	return 1/constants.A_DIM * [Ax_dim, Ay_dim, Az_dim]
end


"""
Overload of getA calculation of the dimensionless vector potential just from the dimensionless position of the particle.
"""
function getA(position, constants)
	x_dim, y_dim, z_dim = constants.L_DIM * position
	rho = getRho(x_dim, y_dim)
	r = getR(rho, z_dim, constants)
	F = getF(r, constants)
	return getA(rho, F, x_dim, y_dim, constants)
end


"""
Calculate dimensionless vector potential in a given dimensionless position, together with its transposed Jacobi matrix.
"""
function getAAndDA_T(position, constants)
	A_as_position_function = ( pos_ -> getA(pos_, constants) )

	A = A_as_position_function(position)
	DA = ForwardDiff.jacobian(A_as_position_function, position)
	return [A, transpose(DA)]
end


"""
Get the dimensionless Hamiltonian value for the current dimensionless particle state.
"""
function getH(position, momentum, constants)
	A = getA(position, constants)
	H = 1/2 * norm(momentum - A)^2
	return H
end


"""
Get the dimensionless position gradient of the dimensionless Hamiltonian.
"""
function getHq(position, momentum, constants)
	A, DA_T = getAAndDA_T(position, constants)
	return -DA_T * (momentum - A)
end


"""
Get the dimensionless momentum gradient of the dimensionless Hamiltonian.
"""
function getHp(position, momentum, constants)
	return momentum - getA(position, constants)
end
