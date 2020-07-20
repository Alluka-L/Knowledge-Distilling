# Knowledge-Distilling
A few experiments on knowledge distilling

## Pretrain Teacher Networks
* Result: ***
* SGD, no weight decay.
* Learning rate adjustment
  * `0.1` for epoch `[1,150]`
  * `0.01` for epoch `[151,250]`
  * `0.001` for epoch `[251,300]`
```
python -m pretrainer --optimizer=sgd --lr=0.1   --start_epoch=1   --n_epoch=150 --model_name=ckpt
python -m pretrainer --optimizer=sgd --lr=0.01  --start_epoch=151 --n_epoch=100 --model_name=ckpt --resume
python -m pretrainer --optimizer=sgd --lr=0.001 --start_epoch=251 --n_epoch=50  --model_name=ckpt --resume
```

### EXP0. Baseline (without Knowledge Distillation)
* Result: 85.01%  85.50%
```
python -m pretrainer --optimizer=adam --lr=0.0001 --start_epoch=1 --n_epoch=300 --model_name=student-scratch --network=studentnet
```

### EXP1. Effect of loss function
* Similar performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 84.99%  85.70%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 84.85%  85.26%
```
