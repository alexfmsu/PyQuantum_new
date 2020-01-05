export PYTHONPATH=/mnt/data/users/dm4/vol12/alexfmsu_2131/software/Python/bin

# ${PYTHONPATH}/python3.7 ~/_scratch/PyQuantum_new/mix_config_mpi.py
# ${PYTHONPATH}/python3.7 mix_g_l.py
${PYTHONPATH}/python3.7 -W 'ignore' mix_g_l_partial.py
# ${PYTHONPATH}/python3.7 mix_config_mpi.py
# ${PYTHONPATH}/python3.7 mix_config_mpi.py
# ${PYTHONPATH}/python3.7 -W 'ignore' $HOME/_scratch/PyQuantum_new/mix_config_mpi.py
# ${PYTHONPATH}/python3.7 -W 'ignore' tt.py
# python -W 'ignore' tt.py
# ${PYTHONPATH}/python3.7 -c 'print(123)'

# sbatch -n 100 -p regular6 --output=reg6.out --error=reg6.err impi ./run.sh