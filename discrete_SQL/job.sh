#!/bin/bash

#SBATCH --job-name=hparam-tuning
#SBATCH --output=hparam_tuning_%j.out
#SBATCH --error=hparam_tuning_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Load any modules and activate your Python environment here
module load python/3.8  

# Clone repo if not present
if [ ! -d "autoregressive-SQL" ]; then
  git clone https://github.com/Angelawork/autoregressive-SQL.git
fi

cd autoregressive-SQL/discrete_SQL/

# install or activate requirements
if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

# log into WandB
export WANDB_API_KEY="a2ff527595e001c6604deef5f2f3a8ed97c08407"
python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"

# Run existing Python script in repo for tuning
python hparam_tuner.py