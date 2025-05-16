# Mimetician
An implementation of the imitation framework **Mimetician** in paper 

_**X2C: A Benchmark Featuring Nuanced Facial Expressions 
for Realistic Humanoid Imitation**_
### Setup


```
conda create -n mimetician python=3.9
conda create -n mimetician python=3.9
conda activate mimetician
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install hydra-core --upgrade
pip install tensorboard
pip install opencv-python   # get frames from video
pip install scipy
pip install transformers
pip install matplotlib
pip install scikit-learn
pip install ffmpeg-python

pip install pytorch-fid  
pip install imageio      
pip install scikit-image
pip install imageio[ffmpeg]
pip install pandas
```
