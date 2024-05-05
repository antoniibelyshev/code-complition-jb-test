# code-complition-jb-test

## How to run

You should start with running the praparation script: "sh run_preparatory_steps.sh". It will install nessecary packages and clone required repos. When installing packages I assumed that standard ml and dl packages like pytorch and sklearn are already installed.

After this, you can run finetune.ipynb which will run all steps including the dataset collection and preprocessing, finetuning and evaluation before and after the tuning. Most of the important code is put into corresponding scripts in utils directory.

I use wandb to log the loss during the training. You have to specify your entity in the line 30 of utils/train.py file, in order to get results to your account.

## Dataset

I decided to parse kotlin code using [kopyt](https://pypi.org/project/kopyt/), because that was the only pyhton package for parsing the kotlin code which I have found. It often makes small mistakes and/or gets errors in parsing, but I was still able to collect an ok dataset of kotlin functions.

## Finetuning

The model is too big to finetune it whole on the gpu that I'm using, so I decided to freeze all layers except for the last 5 layers, which is a sensible technique, at least as far as I know. As I mentioned before, I used wandb to log the loss during the training procedure.

## Evaluation

I took 1000 samples for test part in each dataset, because the generating takes too long for more samples. I used accuracy score, bleu score, and rouge score in order to assess the quality of the model on the test datasets. These metrics were near 0 before finetuning. They did not get much better after the finetuning. However, there is a small increase in rouge score for kotlin dataset (from 0.08 to 0.12). Also the accuracy on the kotlin dataset became non-zero, pointing out that some (small) number of samples got the right answer. However, the metrics on the 