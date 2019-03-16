# ParabolicMongeAmpere_2DMovingMesh
Moving Meshes for 2D PDES using the Monge-Ampere equation

Corresponding example Matlab code from the paper "Adaptive solution to two-dimensional partial differential
equations in curved domains using the Monge-Ampére equation" by Kelsey L. DiPietro and Alan E. Lindsay. 
Accepted for publication in SIAM Journal of Scientific Computing, March 2019. 

The corresponding code runs the example of the Schnakenburg Reaction Diffusion equation on a unit disk, in the
regime corresponding to a splitting spot. Run Schnakenburg_ReactionDiffusion.m for solving and plotting. 

File Dependencies: 
SignedDistance.m (solves the Monge-Ampere mapping for convex domain)
dampednewtons.m (damped newton's method to solve a nonlinear system)
