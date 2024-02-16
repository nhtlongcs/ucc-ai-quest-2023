## Install dependencies

Using conda, recommend mamba for faster solving time

```bash
conda env create -f environment.yml
conda activate ucc
```

## Unzip public.zip in data folder so it has the following structure

```
data/public/img/train
data/public/img/valid
data/public/ann/train
data/public/ann/valid
```

To ensure the data and the environment is setup correctly, run the following command. It should run without error

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/pkg/classics/
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/pkg/transformers/
```

## Usage

Join team using WanDB `https://wandb.ai/ucc-quest-23/ucc-quest-23-public`

More details in `notebooks/train.ipynb`

```
!ucc-train -c <config-file-path> -o <override_arg1>=<value1> <override_arg2>=<value2> ...
```

Some special flags:

-   `global.find_lr=True` : This will find the optimal learning rate for the config file, rerun when have minor change
-   `global.wandb=True`: In the training code include some visualize code using wandb, please not set this value to `False` in the trainning mode.

## Prepare results for submission

After training, the checkpoints are stored in folder `PROJECT_NAME/RUNID/checkpoints`. We need to prepare a file named "results.json" for submission on CodaLab. Use the notebook `notebook/make_submission.ipynb` and replace the checkpoint path

there should be a file `results.zip` generated in the `output` directory. You should be able to submit the file `results.zip` now.
