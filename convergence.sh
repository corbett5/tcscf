#!/bin/bash

for i in {1..9}
do
  srun -n1 ./tests/HartreeFockTest --gtest_filter="*RestrictedClosedShell" -n $i -l 0 -a 1.355 --r1 2000 --r2 1000
done
