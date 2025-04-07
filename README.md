# FLEEG
This is the PyTorch implementation of the FLEEG in our paper:

Rui Liu, Yuanyuan Chen, Anran Li, Yi Ding, Han Yu, Cuntai Guan, "[Aggregating Intrinsic Information to Enhance BCI Performance
through Federated Learning](https://www.sciencedirect.com/science/article/pii/S0893608024000145)", published in the Neural Networks, 2024.

<img src="FLEEG_structure.png" width="800">

If you use this work in your research or project, please cite it as:

```bash
@article{liu2024aggregating,
  title={Aggregating intrinsic information to enhance BCI performance through federated learning},
  author={Liu, Rui and Chen, Yuanyuan and Li, Anran and Ding, Yi and Yu, Han and Guan, Cuntai},
  journal={Neural Networks},
  volume={172},
  pages={106100},
  year={2024},
  publisher={Elsevier}
}
```

## ⚙️ Dataset Preprocessing

- For the KU, SHU, BCI2a and Murat2018 dataset, please download them from the website first, save them to the corresponding folder, and use the provided code to do preprocessing.

- For the Shin, Weibo2014, Cho2017, MunichMI and Schirrmeister2017 datasets, they can be downloaded using the provided code directly, followed by the preprocessing.
  

## 🛠️ Enviroment

- python: 3.10

## 📦 Toolbox

```bash
pip install -r requirements.txt
