# TO_3_Materials
Topology optimization based on NGnet for creation 3-material rotor geometry

This project is a part of the whole optimization project that performs topology optimization based on the normalized Gaussian network for line-start synchronous reluctance machines (3-material optimizations).

The project here only includes the part where the geometry of the rotor is created based on given weighting coefficients w1 and w2. Here, these two coefficients are selected randomly. 

Thus, this part of the project performs the following steps:
-takes weighting coefficients w1/w2
-creates normalized Gassian network (NGnet1/NGnet2)
-creates the geometry of the rotor based on given conditions
-checks the geometry feasibility from the point of view: 
    The iron part must be one connected piece
    There can not be single elements in geometry.

Ing. Iveta Lolová
