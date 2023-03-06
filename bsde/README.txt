
Overview:
	This project implements nonlinear PDE solvers based on deep BSDE (2017) and LSMC algorithm

Components:
	1. Package dynamics
		fbsde.py: Abstract class for the forward and backward SDE.
		european_option.py: Implements the fbsde class for the Black-scholes PDE
		american_option.py: Implements the fbsde class for the American option pricing PDE
		liquidation_1.py: Implements the fbsde class for the HJB-PDE stated in the 
				liquidation with only temporary impact model by Cartea et al. (2015) 
	2. Package solver
		deep_bsde.py: The deep BSDE method in [EHJ17] and [HJE18]
		lsmc.py: The LSMC methods for solving FBSDE
		FBSNNs.py: The FBSNN solver for nonlinear PDEs by Maziar Raissi
	
	3. Package test
		Contains test files and methods for the implementations of different solvers

	4. config.py: 
		the config classes for different FBSDEs and solvers (LSMC, deep BSDE)

	5. engine.py:
		the class that allows the user to solve different FBSDEs using different 
		solvers in one method call




