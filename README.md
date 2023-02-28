# cs224n-project
Project code for Stanford's CS224N

# Installation

```
conda env create -f env.yaml
python -m pip install -e .  
```

# Data

The data for this project comes from [super-natural instructions](https://github.com/allenai/natural-instructions).

# Train
From project root directory:
```
python -m src.model.train --train_tasks=data/02_26_23/train_tasks.txt --test_tasks=data/02_26_23/test_tasks.txt --batch_size=16
```

