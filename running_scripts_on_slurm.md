# Good practises on distributed environment

This file aims at giving some good practise advices on how to run python scripts on shared gpu clusters.

# Transform your python script
TLDR: Replace any hardcoded variables by an input parameters with `argparse`.

## Basic script
Example of a common python script.
<details>
<summary><b>Click to expand the section.</b></summary>

```python
import torch, my_loss # for example purpose
from models import model
from data import load_data # for example purpose


my_model = model(
    n_layers=12,
    n_classes=5,
    dropout=0.1,
)

dataset = load_data("/home/florianlb/data/my_dataset")
dataloader = dataloader(dataset, batch_size=64, shuffle=True)


optimizer = adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in dataloader:
        logits = model(x=batch["input"])
        labels = batch["labels"]
        loss = my_loss(y_pred=logits, y=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
</details>

## Script transformation
How to transform you script using `argparse` to make it easy to use with slurm.
<details>
<summary><b>Click to expand the section.</b></summary>

In order to run your script with Slurm, you should modify it so that things will be easier.
For that we can use `argparse` built-in library.

`argparse` will give access to a `parser` that will read the arguments preceded by `-` or `--` when launching the script.

For example, if I do `python my_script.py --learning_rate 0.001`, then the `parser` will read the attribute `learning_rate` set to 0.001.

**Advantages of `argparse`**
- You can use the **same script for different hyperparameters**.
- All your **hyperparameters are centralized** at the begining.
- General good practise to avoid hardcoded variables.

**Default values**
As illustrated below, you can use default values that won't need to be supplied when launching the scripts.
```python
import torch
from models import model
from data import load_data

# Here is the parser
from argparse import ArgumentParser

parser = ArgumentParser()
# Now just add all our aguments.
# Do not forget to set the dtype, otherwise it will consided everything as a string.
parser.add_argument("--n_layers", dtype=int, default=6)  # specify the dtype.
parser.add_argument("--n_classes", dtype=int, default=5)
parser.add_argument("--dropout", dtype=float, default=0.1)
parser.add_argument("--n_epochs", dtype=int, default=10)
parser.add_argument("--batch_size", dtype=int, default=64)
parser.add_argument("--learning_rate", dtype=float, default=1e-3)
parser.add_argument("--data_path", default="/home/florianlb/data/my_dataset") # do not specify the dtype since it's already a string.


# Retrive the arguments from the command line.
args = parser.parse_args()

my_model = model(
    n_layers=args.n_layers,
    n_classes=args.n_classes,
    dropout=args.dropout,
)

dataset = load_data(args.data_path)
dataloader = dataloader(dataset, batch_size=args.batch_size, shuffle=True)


optimizer = adam(model.parameters(), lr=args.learning_rate)

for epoch in range(args.n_epochs):
    for batch in dataloader:
        logits = model(x=batch["input"])
        labels = batch["labels"]
        loss = my_loss(y_pred=logits, y=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Now you can just launch. Non-specified variables will be set to their default value.
```shell
$ python my_script.py --n_layers 5 --batch_size 128
```
It modifies your hyperparameters efficiently and launch another experiment without touching your training scripts and hence limits errors.
</details>

# Slurm

Using your script with SLURM.

## What is slurm?

<details>
<summary><b>Click to expand the section.</b></summary>

Slurm is an open-source job-scheduler.
Basically, user can ask for resources on which to run their code and slurm dispatches the available resources to the users.

It provides a smooth way to manage resources between uses.

In practice, it means that people have to wait their turn to run a script on the GPU they target (if other people are already using it).

</details>

## Writing a `sbatch` script.
<details>
<summary><b>Click to expand the section.</b></summary>

- Create a file `my_sript.sh` (or whatever name you want).
- `#!/bin/bash` The name of the shell that is going to run the program.
- `#SBATCH` the lines begining with `#SBATCH --<param>` specify parameters for slurm. There are many, below is an example for the most important ones.
- Your script, just as you would launch it in the terminal.
-  You can break lines between arguments with ` \ `, as illustrated below (more readable).

```bash
#!/bin/bash
#SBATCH --partition=<your target partition>
#SBATCH --job-name=<your job name>
#SBATCH --nodelist=<your target node>
#SBATCH --nodes=<number of wanted node>
#SBATCH --time=<d-h:m:s timelimit for the job>
#SBATCH --output=<path/to/output_file>

python my_script \
    --n_layers 6 \
    --n_classes 5 \
    --dropout 0.1 \
    --n_epochs 10 
```
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
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --output=training_punk.out
...
```

I want:
- 2 GPUs on Led
- For 1 day and 5h
```bash
#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=training_model
#SBATCH --nodelist=led
#SBATCH --nodes=2
#SBATCH --time=1-5:00:00
#SBATCH --output=training_hard.out
...
```

## Launch a job
You only need to run:
```shell
$ sbatch my_script.sh
```
## Monitoring your jobs
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
$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             67164      hard ssm_trad florianl PD       0:00      1 (Resources)
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