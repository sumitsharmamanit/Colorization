# Conditional Colorization
-----------------------

Model is based on UNet architecture. It accepts [ l . ab_gray , cue ] as the input.
The 4 channels consist 3 color channel (e.g. Lab format L and ab channel) and a cue image in one channel.   
The cue channel provides 0.1% true pixels and helps model to train efficiently.

The model outputs a 3 channels tensor which is converted into an image of ".png" format.    
The data is augmented prior to the training ( Flip horizontal , Rotate 180 ) to over sample the training dataset.    

------------------------
Please refer to folder.png for the folder structure.

Make sure you have cuda enabled machine. I used AWS Sagemaker to get ready made enviroment.

Instruction to run the code:

1. Check out the code 
   `git clone https://github.com/sumitsharmamanit/Colorization.git`
2. `pip install requirments.txt`
3. Execute training.ipynb notebook file
4. for prediction, use predict.ipynb. There are two modes: 1. bulk prediction 2. single image prediction
5. place the best weight under checkpoints folder
-----------------------
## Required Library
-----------------------
matplotlib==3.5.1
numpy==1.22.1
opencv-python==4.5.5.62
Pillow==9.0.0
pyparsing==3.0.6
torch==1.7.1
torchvision==0.11.2
tqdm==4.62.3
-----------------


Best Weight file
```
https://drive.google.com/file/d/1MpPIH1KbQSNcNo3UQYjuBPV7JSrnaF_e/view?usp=sharing
```
-----------------
