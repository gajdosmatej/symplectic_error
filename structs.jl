struct Constants
	R
	a
	B0
	m
	Q
	L_DIM
	T_DIM
	P_DIM
	H_DIM
	A_DIM

	function Constants(R, a, B0, m, Q)
		# Define characteristic dimensions
		L_DIM = 2*pi*R	# one toroidal cycle
		T_DIM = m/(Q*B0) # set the maximal gyrofrequency to 1 
		P_DIM = m*L_DIM / T_DIM
		H_DIM = P_DIM^2 / m
		A_DIM = P_DIM / Q

		new(R, a, B0, m, Q, L_DIM, T_DIM, P_DIM, H_DIM, A_DIM)
	end
end

struct InitialConditions
	q0
	p0
end
