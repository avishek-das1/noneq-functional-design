# Nonequilibrium design strategies for functional colloidal assemblies
This repository contains a working code and the data for reproducing the corresponding paper [arXiv:2209.11538](https://arxiv.org/abs/2209.11538).

Folders fig{1,2,3,4} and sifig{1,2,3,4} each contain the data and python scripts needed to reproduce the figures.

optimize_current.f90 is an example code for optimizing probability current from the N=7 C2 nanocluster to the second C3v nanocluster.
To run:

mpifort -O3 optimize_current.f90 -o optimize
mpirun -np n ./optimize

where n should be >=2, as the first thread is always dedicated to collecting and averaging data from all other threads.
