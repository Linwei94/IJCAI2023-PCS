# [Calibrating a Deep Neural Network with Its Predecessors](https://arxiv.org/abs/2302.06245)
Good news! We provided a [colab playground](https://colab.research.google.com/drive/1TjwzG962eyOF51zzqlWLwv3Wq-lgMHZM?usp=sharing) where you can build your own **Predecessors Combination** and see its perforamnce!

> A PyTorch implementation for _Calibrating a Deep Neural Network with Its Predecessors_. Licensed under the Apache License, Version 2.0.
> 
> **This project is based on [Focal Calibration](https://github.com/torrvision/focal_calibration).**
> 


## Requirements
```
Python >= 3.7.5, PyTorch == 1.3.1, torchvision == 0.4.2
```

## Instruction

* `train_search.py`: search for architectures;
```
python train_search.py --epochs=350 --seed=1 --warm_up_population=100 --gumbel_scale=1e-1 --arch_learning_rate=1e-2 --memory_size=100 --predictor_warm_up=500 --dataset_name=cifar10 --model_name=resnet50 --ftlr=1e-3
```
* `train.py`: train models;
```
python train.py --model resnet50 --dataset-root DATASET_DIR --loss cross_entropy
```
* `test_combination.py`: test predecessor combinations;
```
python test_combination.py --dataset cifar10 --model resnet50 --tune_epoch 1 --combination 186,313,299,139,189 -log --weight_folder WEIGHT_DIR --lr=1e-4
```


## Citations

If you are using the code/data in this repo, please consider citing:

```
      @inproceedings{linwei2023calibrating,
        title     = {Calibrating a Deep Neural Network with Its Predecessors}, 
        author    = {Linwei Tao and Minjing Dong and Daochang Liu and Changming Sun and Chang Xu},
        booktitle = {Proceedings of the Thirtieth International Joint Conference on
                     Artificial Intelligence, {IJCAI-23}},
        publisher = {International Joint Conferences on Artificial Intelligence Organization},
        note      = {Main Track}
      }
```
