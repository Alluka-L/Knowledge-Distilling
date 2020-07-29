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
* Result: 85.56%
```
python -m pretrainer --optimizer=adam --lr=0.0001 --start_epoch=1 --n_epoch=300 --model_name=student-scratch --network=studentnet
```

### EXP1. Effect of loss function
* Similar performance.
```
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=cse # 85.70%
python -m trainer --T=1.0 --alpha=1.0 --kd_mode=mse # 85.71%
```

### EXP2. Effect of model
```
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=cse # 85.49%
python -m trainer --T=1.0 --alpha=0.5 --kd_mode=mse # 85.46%
```

### EXP3. Effect of Temperature Scaling
* Higher the temperature, better the performance. Consistent results with the paper.
```
python -m trainer --T=2.0   --alpha=1.0 --kd_mode=cse # 86.2%
python -m trainer --T=2.0   --alpha=1.0 --kd_mode=mse # 85.53%
python -m trainer --T=4.0   --alpha=1.0 --kd_mode=cse # 86.69%
python -m trainer --T=4.0   --alpha=1.0 --kd_mode=mse # 86.09%
python -m trainer --T=8.0   --alpha=1.0 --kd_mode=cse # 86.24%
python -m trainer --T=8.0   --alpha=1.0 --kd_mode=mse # 86.89%
python -m trainer --T=12.0  --alpha=1.0 --kd_mode=cse # 86.01%
python -m trainer --T=12.0  --alpha=1.0 --kd_mode=mse # 86.56%
python -m trainer --T=16.0  --alpha=1.0 --kd_mode=cse # 86.41%
python -m trainer --T=16.0  --alpha=1.0 --kd_mode=mse # 86.46%
```

### EXP4. More Alpha Tuning
```
python -m trainer --T=8.0  --alpha=0.1 --kd_mode=cse # 86.52%
python -m trainer --T=8.0  --alpha=0.3 --kd_mode=cse # 86.97%
python -m trainer --T=8.0  --alpha=0.5 --kd_mode=cse # 86.86%
python -m trainer --T=8.0  --alpha=0.7 --kd_mode=cse # 86.79%
python -m trainer --T=8.0  --alpha=0.9 --kd_mode=cse # 86.87%
python -m trainer --T=8.0  --alpha=0.1 --kd_mode=mse # 85.39%
python -m trainer --T=8.0  --alpha=0.3 --kd_mode=mse # 85.65%
python -m trainer --T=8.0  --alpha=0.5 --kd_mode=mse # 85.93%
python -m trainer --T=8.0  --alpha=0.7 --kd_mode=mse # 85.79%
python -m trainer --T=8.0  --alpha=0.9 --kd_mode=mse #
python -m trainer --T=16.0 --alpha=0.5 --kd_mode=cse #
python -m trainer --T=16.0 --alpha=0.7 --kd_mode=cse #
python -m trainer --T=16.0 --alpha=0.9 --kd_mode=cse #
```
