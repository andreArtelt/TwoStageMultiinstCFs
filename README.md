# A Two-Stage Algorithm for Cost-Efficient Multi-instance Counterfactual Explanations

This repository contains the implementation of the paper [A Two-Stage Algorithm for Cost-Efficient Multi-instance Counterfactual Explanations](paper.pdf) by André Artelt and Andreas Gregoriades

## Abstract

Counterfactual explanations constitute among the most popular methods for analyzing the predictions of black-box systems since they can recommend cost-efficient and actionable changes to the input to turn an undesired system's output into a desired output. While most existing methods explain a single instance only, several real-world use cases exist, such as customer satisfaction, where one is interested in a single counterfactual explanation that simultaneously explains multiple instances.

In this work, we propose a flexible two-stage algorithm for finding groups of instances along with cost-efficient multi-instance counterfactual explanations. In particular, the aspect of finding such groups has been mostly ignored in the literature so far.

## Details

### Data

The data sets used in this work are stored in [Implementation/data/](Implementation/data/). Note that many of these .csv files in the data folder were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data.

### Experiments

All experiments are implemented in [Implementation/experiments.py](Implementation/experiments.py).

Our proposed evolutionary algorithm for computing multi-instance counterfactual explanations is implemented in [Implementation/ours](Implementation/ours). The methods proposed by Warren et al. and Kanamori et al. are implemented in [Implementation/warren](Implementation/warren) and [Implementation/kanamori](Implementation/kanamori) which was taken from their [repository](https://github.com/kelicht/cet) -- note that for the method proposed by Kanamori et al. the [IBM CPLEX](https://www.ibm.com/de-de/products/ilog-cplex-optimization-studio) solver is required. 

## Requirements

- Python 3.8
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt)

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

You can cite the version on [arXiv](https://arxiv.org/abs/2403.01221).
