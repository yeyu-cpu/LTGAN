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
find model.py in clip，comment line 235,237,238  
```
# x = self.ln_post(x[:, 0, :])  

# if self.proj is not None:  
#     x = x @ self.proj
```
