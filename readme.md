# EMultiplicative Logit Adjustment Approximates Neural-Collapse-Aware Decision Boundary Adjustment (ICLR2025)

[[ArXiV]](https://arxiv.org/abs/2409.17582)
[[OpenReview]](https://openreview.net/forum?id=II81zQUS1x)

## Code Description
This repository comprises code for LTR. Primarily, one can conduct training (or logit adjustment) and assess the model by executing first.py and second.py at the top level.

- `first.py`: Code for the training.
- `second.py`: Code for the logit adjustment or 1vs1 adjuster.

This repository is based on the [paper](https://arxiv.org/abs/2203.14197) and [repository](https://github.com/ShadeAlsha/LTR-weight-balancing). Please refer to them as well.

## How to Execute
1. Setup environment. Refer to [Requirement](#Requirement).
2. Prepare datasets. If you want to experiment with CIFAR, this program automatically download them to the `datasets` directory. 
If you want to use ImageNet or other datasets, place them in the `datasets` folder beforehand. The location can be modified in `utils/conf.py`.
3. Execute. Run the program by executing first.py or second.py. You can reproduce experiments detailed in our paper by specifying the JSON file path in the `jsons` directory. For instance:
```python
python first.py --seeds 0 --json_path "./jsons/Cifar100/first.json"
```
```python
python second.py --seeds 0 --json_path "./jsons/Cifar100/second.json"
```
4. Check the results by examining the standard output and the `exp` folder.

## Requirement

1. Install pip, Python, and PyTorch. Our environment specifications are depicted below. Other versions might function, albeit reproducibility of results cannot be guaranteed.
   - Python version: Python 3.6.8
   - PyTorch verion: 1.10.1+cu113
2. Install the libraries via pip using the following command.
```
pip install -U scikit-image pandas seaborn ipykernel scikit-learn tensorboard
```




## Citation
If you find our model or methodology beneficial, kindly cite our work:

    @inproceedings{
      hasegawa2024exploring,
      title={Exploring Weight Balancing on Long-Tailed Recognition Problem},
      author={Naoya Hasegawa and Issei Sato},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=JsnR0YO4Fq}
    }
