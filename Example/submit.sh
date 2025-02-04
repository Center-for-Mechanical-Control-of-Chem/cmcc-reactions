#!/bin/bash

#SBATCH -t 24:00:00             #Time for the job to run
#SBATCH -J GSM                 #Name of the job appears on the queue
#SBATCH -o GSM.out             #Name of the output log file
#SBATCH -e GSM.err             #Log file for any errors
#SBATCH --job-name=GSM        
#SBATCH --time=24:00:00          # Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                      #Request tasks
#SBATCH --cpus-per-task=4     #Request CPUs per task
#SBATCH --mem-per-cpu=7000

set OMP_NUM_THREADS = 4
module load Anaconda3/2021.05
source activate aimnet2
python test.py -xyzfile conf.xyz -isomers isomer.txt -mode SE_GSM -package ase -num_nodes 20 -mp_cores 4 -constraints_file constraints.txt
convert_conf.py
python test.py -xyzfile conf_continue.xyz -mode DE_GSM -package ase -mp_cores 4 -constraints_file constraints.txt
