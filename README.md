# cs224n-project
Project code for Stanford's CS224N

# Installation

```
conda env create -f env.yml
python -m pip install -e .  
```

# Data

The data for this project comes from [super-natural instructions](https://github.com/allenai/natural-instructions).

# Train/Test split
```
cd data
```

Edit `test_categories.txt` and `train_categories.txt` with the desired categories.

Then run:
```
python create_train_test_splits.py --output_dir=<name>
```
# Sampling
Either use `--sample_num` or `--sample_rate`
```
cd data
python sample_tasks.py --csv=same_cluster/test.csv --sample_num=1000
```

# Train
From project root directory:
```
python -m src.model.train --train_tasks=data/overfit/train_tasks.txt --test_tasks=data/overfit/test_tasks.txt --batch_size=16
```


