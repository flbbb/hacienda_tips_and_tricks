# Writing Python script.
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
For that we can use `argparse` built-in library to input parameters.

`argparse` will give access to a `parser` that will read the arguments preceded by `-` or `--` when launching the script.

For example, if I do `python my_train_script.py --learning_rate 0.001`, then the `parser` will read the attribute `learning_rate` set to 0.001.

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
$ python my_train_script.py --n_layers 5 --batch_size 128
```
It modifies your hyperparameters efficiently and launch another experiment without touching your training scripts and hence limits errors.
</details>
