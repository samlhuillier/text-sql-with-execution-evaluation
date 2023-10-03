# C3SQL
The code for the paper C3: Zero-shot Text-to-SQL with ChatGPT ([https://arxiv.org/abs/2307.07306](https://arxiv.org/abs/2307.07306))

## Prepare Spider Data

Download [spider data](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ) and database (only spider original database right now) and then unzip them:

```shell
mkdir data 
unzip spider.zip 
mv spider/database . 
mv spider data
```

### Run evaluation:
(Install the python dependencies it complains about)
```
python third_party/test-suite-sql-eval/evaluation.py --gold spider-create-context-intersect/spider_create_context_gold-568.sql --pred llama7b-predicted2.txt --db /Users/sam/Downloads/spider/database --table spider/tables.json --etype all
```
Replace ```llama7b-predicted2.txt``` with your generated file. Bear in mind that, ```spider-create-context-intersect/spider_create_context_gold-568.sql``` are the ground truth values for my custom subset of [spider](https://huggingface.co/datasets/samlhuillier/sql-create-context-spider-intersect). Use ```spider/dev_gold.sql``` for the full spider ground truth values. 
