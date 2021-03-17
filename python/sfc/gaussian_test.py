from multi_agent_kinetics import projections, kernels, viz
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

for i in range(1):
    kernels.gaussian_function(0.5)