# X2CNet
This repository is the official implementation of the imitation framework **X2CNet** in paper 

_**X2C: A Dataset Featuring Nuanced Facial Expressions 
for Realistic Humanoid Imitation**_ 


![Alt text](docs/static/images/imitation_framework-nips.png)

## Getting Started 🏁
### 1. Clone the code and prepare the environment 🛠️

```bash
git clone git@github.com:lipzh5/X2CNet.git
cd X2CNet

# create env using conda
conda create -n x2cnet python=3.9
conda activate x2cnet
# for cuda 12.1
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

To install requirements:

```setup
pip install -r requirements.txt
```


## Training
```train
python main.py train.batch_size=128 train.num_workers=16 train.num_epochs=100 train.lr=1e-3
```

## Evaluation
```eval
python main.py do_eval=True train.batch_size=128 train.num_workers=16 train.save_model_path=path/to/save_folder
```

## Pre-trained Models
You can download pre-trained models here:

## Real-world Inference Results
![Alt text](docs/static/images/inference_example3_160.png)

## Contributing
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.


## Citation 💖
If you find <strong>X2C</strong> or <strong>X2CNet</strong> useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{li2025x2c, title={X2C: A Dataset Featuring Nuanced Facial Expressions for Realistic Humanoid Imitation}, 
author={Li, Peizhen and Cao, Longbing and Wu, Xiao-Ming and Yang, Runze and Yu, Xiaohan}, journal={arXiv preprint arXiv:2505.11146}, 
year={2025} }
```

*Long live in arXiv.*

