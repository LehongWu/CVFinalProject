# PKU 2023 Computer Vision Final Project - MaskGIT

## Code Files & Folders 
main.py: run the code  
engine_train/test.py: contain the training/testing function  
model: network architectures  
utils: tools (visualize/metrics/...)  
|-- metrics.py(calculating FID): require package **pytorch_fid**  
feeder: dataloader  
maskgit_reference_code: code framework provided by teacher(useless)  

## Other Files & Folders
will be created manually or by running the code  

datasets(ignored): should be like this   
|-- tiny-imagenet-200  
|&emsp; |-- train  
|&emsp; |-- test  
|&emsp; |-- val(useless)  
|&emsp; |-- some labels(.txt)(not been used so far)  
|-- mnist  
|-- cifar10  
exp: tensorboard outputs  
ckpt: checkpoints  
visualize: visualization(testing) results  

## Assets
Provide pre-trained checkpoints.
Download pretrained tokenizer [here.](https://drive.google.com/drive/folders/1cKtNKXvXgCjHyubQT3qI35A0ap5gTYi5)
