## FlowNLG 
![flow](git-resources/flow-symbol.png)
Enriched variant of [DeepNLG](https://github.com/ThiagoCF05/DeepNLG), introducing information flow hints. 

### Prerequistes
It is recommended to run the code in a docker container or a designated conda envronment.

* clone the repository. 
* install requirements with `pip install -r requirements.txt`. (possibly change `torch` version depending on your version of `cuda` )

The code is compatible with `cuda`, and it is recommended to run it on a `cuda` machine, but runs on cpu as well.

### Generating enriched data from DeepNLG
Enriched dataset is located in [`data`](data) folder by default.
Running [`DatasetGenerator.py`](DatasetGenerator.py) will create the enriched dataset from DeepNLG located in [`DeepNLG_data`](DeepNLG_data)

### Evaluating text generated from FlowNLG entries
To generate text from FlowNLG and measure BLEU metric with respect to references run [`EvalOracle.py`](EvalOracle.py) 