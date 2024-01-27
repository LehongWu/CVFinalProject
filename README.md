# PKU 2023 Computer Vision Final Project - MaskGIT

## Code Files & Folders 
main.py: run the code  

engine_train/test.py: contain the training/testing function of VQVAE

main_transformer.py: run the code of the generation model

engine_train/test_transformer.py: contain the training/testing function of MASKGIT

main_classify.py: run the code of the classification model  

engine_train/test_classify.py: contain the training/testing function of the classification model

model: network architectures  
- VQVAE_ver3.py and VQVAE4transformer_ver3.py are for TinyImageNet Dataset due to some differences in VQVAE. Please note that the models in these two files are not compatible with the version without suffix. You need to manually modify the relevant import code when using TinyImageNet.

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
