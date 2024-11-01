# Enhancing Makeup Transfer Robustness under Varied Lighting Conditions with Lighting Transfer GAN
## Getting Started
Install Pytorch and torchvision(torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1)  
Install the following libraries  
```bash  
pip install git+https://github.com/openai/CLIP.git
```
```  
pip install opencv-python matplotlib dlib fvcore
```    
Find model.py in clip，comment line 235,237,238  
```
# x = self.ln_post(x[:, 0, :])  

# if self.proj is not None:  
#     x = x @ self.proj
```
Pretrained model can be downloaded from [weights](https://pan.baidu.com/s/1K6CCFJapkdvAYAeISlvAvw?pwd=j0ka ), put G.pth into ./ckpts  
Testing data can be downloaded from [data](https://pan.baidu.com/s/1Ahokgl8AF_-ZGQMqi1CwRQ?pwd=a260)
## Test  
to test our model, run:  
```
python scripts/demo.py
```  
