# IGMTF: Instance-wise Graph-based Framework for Multivariate Time Series Forecasting
The source code and data of the paper "[Instance-wise Graph-based Framework for Multivariate Time Series Forecasting](https://arxiv.org/abs/2109.06489)". 
![image](https://user-images.githubusercontent.com/25242325/139006981-fcded966-daf1-4697-955e-a1b4fdf4e08a.png)

## Requirements
The framework is implemented using python3 with dependencies specified in [requirements.txt](https://github.com/Wentao-Xu/IGMTF/blob/main/requirements.txt).

## Datasets
### Overall information of datasets

| Datasets      | Variants | Timesteps | Granularity | Time | Task Type   |
| ------------- | -------- | --------- | ----------- | ---------- | ----------- |
| Traffic       | 862      | 17,544    | 1hour       |2015 to 2016| Single-step |
| Electricity   | 321      | 26,304    | 1hour       |2012 to 2014| Single-step |
| Exchange-Rate | 8        | 7,588     | 1hour       |1990 to 2016| Single-step |

## Reproduce the results
![image](https://user-images.githubusercontent.com/25242325/139007069-798c714c-ba8f-48c7-8838-6b15512bc602.png)

```
git clone https://github.com/Wentao-Xu/IGMTF.git
cd IGMTF
tar -zxvf data.tar.gz
mkdir model
```


* Traffic 

```
# Horizon 3
python learn.py --save ./model/model-traffic-3.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 3 --hidden_size 256 --hidden_batch_size 64 --k_day 30 --n_neighbor 20

# Horizon 6
python learn.py --save ./model/model-traffic-6.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 6 --hidden_size 256 --hidden_batch_size 64 --k_day 5 --n_neighbor 30

# Horizon 12
python learn.py --save ./model/model-traffic-12.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 12 --hidden_size 256 --hidden_batch_size 64 --k_day 10 --n_neighbor 30

# Horizon 24
python learn.py --save ./model/model-traffic-24.pt --data ./data/traffic.txt --num_nodes 862 --epoch 100 --horizon 24 --hidden_size 256 --hidden_batch_size 64 --k_day 3 --n_neighbor 10
```

* Electricity

```
# Horizon 3
python learn.py --save ./model/model-electricity-3.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 3 --hidden_size 512 --k_day 5 --n_neighbor 20

# Horizon 6
python learn.py --save ./model/model-electricity-6.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 6 --hidden_size 512 --k_day 3 --n_neighbor 3

# Horizon 12
python learn.py --save ./model/model-electricity-12.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 12 --hidden_size 512 --k_day 10 --n_neighbor 5

# Horizon 24
python learn.py --save ./model/model-electricity-24.pt --data ./data/electricity.txt --num_nodes 321 --epoch 100 --horizon 24 --hidden_size 512 --k_day 5 --n_neighbor 20
```

* Exchange-Rate

```
# Horizon 3
python learn.py --save ./model/model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --epoch 100 --horizon 3 --hidden_size 512 --batch_size 16 --k_day 10 --n_neighbor 10

# Horizon 6
python learn.py --save ./model/model-exchange-6.pt --data ./data/exchange_rate.txt --num_nodes 8 --epoch 100 --horizon 6 --hidden_size 512 --batch_size 16 --k_day 10 --n_neighbor 10

# Horizon 12
python learn.py --save ./model/model-exchange-12.pt --data ./data/exchange_rate.txt --num_nodes 8 --epoch 100 --horizon 12 --hidden_size 512 --batch_size 16 --k_day 10 --n_neighbor 10

# Horizon 24
python learn.py --save ./model/model-exchange-24.pt --data ./data/exchange_rate.txt --num_nodes 8 --epoch 100 --horizon 24 --hidden_size 512 --batch_size 16 --k_day 10 --n_neighbor 10
```
