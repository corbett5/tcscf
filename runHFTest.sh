#!/bin/bash
 
#SBATCH -N 1
#SBATCH -n 1

./tests/HartreeFockTest --gtest_filter="*Helium_Ochi"