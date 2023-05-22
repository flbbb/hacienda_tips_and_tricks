# Good practises on distributed environment

This file aims at giving some good practice advices on how to run python scripts on shared gpu clusters.


# Slurm

Using your script with SLURM.

## What is slurm?

<details>
<summary><b>Click to expand the section.</b></summary>

Slurm is an open-source job-scheduler.
Basically, user can ask for resources on which to run their code and slurm dispatches the available resources to the users.

It provides a smooth way to manage resources between users.

In practice, it means that people have to wait their turn to run a script on the GPU they target (if other people are already using it).

</details>

## How can I launch my scripts on a GPU?

<details>
<summary><b>Click to expand the section.</b></summary>

We suppose that you have a fully working python script `my_train_script.py`.

To execute it on a GPU there are two ways.
- `sbatch` to launch the script in the background.
- `srun` where you connect to an environment that has access to the GPU and where you can launch your scripts interactively. Should be used mainly for debuging purposes.

### SBATCH

Just below is a an example sbatch script. Next section provides real examples.

```bash
#!/bin/bash
#SBATCH --partition=<your target partition>
#SBATCH --nodelist=<your target nodes>
#SBATCH --gpus=<number of wanted gpus>
#SBATCH --job-name=<your job name>
#SBATCH --time=<d-h:m:s timelimit for the job>
#SBATCH --output=<path/to/output_file>

srun python my_train_script.py \
    --n_layers 6 \
    --n_classes 5 \
    --dropout 0.1 \
    --n_epochs 10 
```
- Create a file `my_sript.sh` (or whatever name you want).
- `#!/bin/bash` The name of the shell that is going to run the program.
- `#SBATCH` the lines begining with `#SBATCH --<param>` specify parameters for slurm. There are many, below is an example for the most important ones.
- `srun` + your instruction, just as you would launch it in the terminal. For example if you usually do `python my_train_script.py`, it becomes `srun python my_train_script.py`.
-  You can break lines between arguments with ` \ `, as illustrated (more readable).

**How to choose the GPUs?**
The GPUs are organized into partition and nodes.
A node contains several GPUs.
A partition contains several nodes.

You specify the ones you want using the SBATCH parameters as shown above.

## Launch a job
You only need to run:
```shell
$ sbatch my_script.sh
```

## Did it work?
You just launched `sbatch my_script.sh`.

- Run `$ squeue -u $USER` you should see your name (not an empy list).

All the script output is stored in the file that you supplied in `#SBATCH --output=my_output.out`.
You can print the file content with `cat my_output.out`.

</details>

## Hands-on examples
<details>
<summary><b>Click to expand the section.</b></summary>

I want:
- 1 GPU on Punk
- For 2h
```bash
#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=training_model
#SBATCH --nodelist=punk
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --output=training_punk.out
srun python my_train_script.py
```

I want:
- 2 GPUs on Led
- For 1 day and 5h
```bash
#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=training_model
#SBATCH --nodelist=led
#SBATCH --gpus=2
#SBATCH --time=1-5:00:00
#SBATCH --output=training_led.out
srun python my_train_script.py
```

I want:
- 3 GPUs in total, on Thin and Lizzy
- For 8h
```bash
#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=training_model
#SBATCH --nodelist=thin,lizzy
#SBATCH --gpus=3
#SBATCH --time=8:00:00
#SBATCH --output=training_thin_lizzy.out
srun python my_train_script.py
```

</details>

## Monitoring your jobs
<details>
<summary><b>Click to expand the section.</b></summary>

- `squeue` will print all running jobs on the cluster.
```shell
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             67067 electroni     bash falissar  R    3:52:58      1 punk
             66673 electroni     bash     rame  R 1-03:18:03      1 daft
             67087     funky run_init    migus  R    2:56:25      1 rodgers
             67088     funky run_init    migus  R    2:56:25      1 rodgers
```

- Get all my jobs:
```shell
$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             67164      hard ssm_trad florianl PD       0:00      1 (Resources)
```
- Get all the jobs of a given user:
```shell
$ squeue -u <user_name>
```
- Get all jobs on a given partition (e.g. on `hard`):
```shell
$ squeue -p hard
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             67164      hard ssm_trad florianl PD       0:00      1 (Resources)
             66308      hard      inr   kassai  R 2-14:10:44      1 lizzy
             67149      hard    t5xxl erbacher  R      47:23      1 zeppelin
             67125      hard     bash  luiggit  R    1:29:52      1 thin
             67048      hard      inr  serrano  R    7:21:51      1 top
```

-  `squeue -l` more info on the time limit (can be combined with any of the above example).

</details>
