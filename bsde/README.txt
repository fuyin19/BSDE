config.py:
    Contains config classes for different FBSDEs and models (LSMC, deep BSDE)

FBSDE.py:
	Contains abstract classes for FBSDE

LSMC.py:
	Contains the LSMC methods for solving FBSDE

LSMC_engine.py:
    Use LSMC method to solve FBSDEs with different configs at the same time

deep_bsde.py:
    The deep BSDE method in [EHJ17] and [HJE18]

Liquidation1.py:
	Should contain the LSMC implementation of the PDE on P140 of the Algo and High-frequency trading book. Will be completed after finding the FBSDE representation of the PDE.

european_option.py: 
	Contains pricing methods using BS formula, Monte-Carlo for FSDE, and LSMC for FBSDE (BS equations)

american_option.py:
    Contains pricing methods for American option using BS formula (r=0, d=0), PDE (variational inequality), and LSMC
    (currently only implemented for American vanilla option)


