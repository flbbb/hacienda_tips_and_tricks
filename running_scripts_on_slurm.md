# Good practises on distributed environment

This file aims at giving some good practice advices on how to run python scripts on shared gpu clusters.

# Transform your python script
TLDR: Replace any hardcoded variables by an input parameters with `argparse`.

## Basic script
Example of a common python script.
<details>
<summary><b>Click to expand the section.</b></summary>

```python 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

def MyModel(nn.Module):
    def __init__(self, d_model, n_classes):
        self.linear = nn.Linear(d_model, n_classes)
    
    def forward(x):
        x = x.flatten(start_dim=1)
        logits = self.linear(x)
        return logits

# MNIST is 28 * 28 and 10 classes.
model = MyModel(d_model=28 * 28, n_classes=10)

dataset = datasets.MNIST(
    root='./data',
    download=True,
    transform=torchvision.transform.ToTensor()
)
dataloader = DataLoader(dataset, batch_size=32)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(10):
    for x, labels in dataloader:
        logits = model(x)
        loss = loss_fn(y_pred=logits, y=labels)
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

**`if __name__ == "__main__"`**
This is a good practice tips. It executes the code beneath the conditional statement only if it has been launched from the command line.
It is useful because you can now import your class `MyModel` into other files without executing all the code below.
```python 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from argparse import ArgumentParser


def MyModel(nn.Module):
    def __init__(self, d_model, n_classes):
        self.linear = nn.Linear(d_model, n_classes)
    
    def forward(x):
        x = x.flatten(start_dim=1)
        logits = self.linear(x)
        return logits


# __name__ == "__main__" checks if the script is called from the command line.
if __name__ == "__main__":
    parser = ArgumentParser()
    # Now just add all our aguments.
    # Do not forget to set the dtype, otherwise it will consided everything as a string.
    parser.add_argument("--n_classes", type=int, default=5) # dtype and default value
    parser.add_argument("--model_dim", type=int, default=784)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    model = MyModel(d_model=args.model_dim, n_classes=args.n_classes)

    dataset = datasets.MNIST(
        root='./data',
        download=True,
        transform=torchvision.transform.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(args.n_epochs):
        for x, labels in dataloader:
            logits = model(x)
            loss = loss_fn(y_pred=logits, y=labels)
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

It provides a smooth way to manage resources between users.

In practice, it means that people have to wait their turn to run a script on the GPU they target (if other people are already using it).

</details>

## How can I launch my scripts on a GPU?

<details>
<summary><b>Click to expand the section.</b></summary>

We suppose that you have a fully working python script `my_script.py`.

To execute it on a GPU there are two ways.
- `sbatch` to launch the script in the background.
- `srun` where you connect to an environment that has access to the GPU and where you can launch your scripts interactively. Should be used mainly for debuging purposes.

### SBATCH

Just below is a an example sbatch script.

```bash
#!/bin/bash
#SBATCH --partition=<your target partition>
#SBATCH --nodelist=<your target nodes>
#SBATCH --gpus=<number of wanted gpus>
#SBATCH --job-name=<your job name>
#SBATCH --time=<d-h:m:s timelimit for the job>
#SBATCH --output=<path/to/output_file>

srun python my_training_script.py \
    --n_layers 6 \
    --n_classes 5 \
    --dropout 0.1 \
    --n_epochs 10 
```
- Create a file `my_sript.sh` (or whatever name you want).
- `#!/bin/bash` The name of the shell that is going to run the program.
- `#SBATCH` the lines begining with `#SBATCH --<param>` specify parameters for slurm. There are many, below is an example for the most important ones.
- `srun` + your instruction, just as you would launch it in the terminal. For example if you usually do `python my_training_script.py`, it becomes `srun python my_training_script.py`.
-  You can break lines between arguments with ` \ `, as illustrated (more readable).

**How to choose the GPUs?**
The GPUs are organized into partition and nodes.
A node contains several GPUs.
A partition contains several nodes.

You specify the ones you want using the SBATCH parameters as shown below.
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

## Launch a job
You only need to run:
```shell
$ sbatch my_script.sh
```

## Did it work?
You just launched `sbatch my_script.sh`.

All the script output is stored in the file that you supplied in `#SBATCH --output=my_output.out`.
You can print the file content with `cat my_output.out`.

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
