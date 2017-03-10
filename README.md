# Loss-Sensitive GAN

Author: Guo-Jun Qi, Date: 1/9/2017, most recent update: 2/8/2017

Questions about the source codes can be directed to Dr. Guo-Jun Qi at guojunq@gmail.com.  All copyrights reserved.

Please cite the following paper when referring to the following algorithms (LS-GAN and CLS-GAN)

**Guo-Jn Qi. Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities. arXiv:1701.06264 [[pdf](https://arxiv.org/abs/1701.06264)]**
  
We are keeping updating this repository of source codes, and more results and algorithms will be released soon.

- We now have a new project generalizing LS-GAN to a more general form, called **Generalized LS-GAN (GLS-GAN)**. It unifies Wasserstein GAN as well as this LS-GAN in an integrated framework, along with a super class of GLS-GANs including various new GAN members to explore. Check the github project at https://github.com/guojunq/glsgan, and the Appendix D of the newly updated prepint 

- We also prepare **an incomplete map of GANs** at http://www.cs.ucf.edu/~gqi/GANs.htm, which plots the terrority of basic GANs in our view. We will update the map as more GAN territories are found.

## Notes on setting the hyperparameters
- **gamma:** this is the coefficient for loss-minimization term (the first term in the objective for optimizing L_\theta). Although in theory, this term can be discarded without affecting the consistency of densities between real and generated samples, we found setting a small value to gamma would  make the generated images more smooth and visually plausible.  This is perphas because allowing a small deviation from the true data density could avoid the areas with abrustly changing densities, thus focusing on more global manifold structure of real images. 
- **lambda:** the scale of margin. This controls the desired margins between real and fake samples.  However, we found this is a hyperparameter not very sensitive to the generation performance.


## LS-GAN (without conditions)

### For celebA dataset
1.Setup and download dataset 

```bash
`mkdir celebA; cd celebA
```

Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".

```bash
unzip img_align_celeba.zip; cd ..
DATA_ROOT=celebA th data/crop_celebA.lua
```

2.Training the LS-GAN

```bash
DATA_ROOT=celebA dataset=folder th lsgan.lua
```

### For LSUN bedroom dataset 

1. Please download bedroom_train_lmdb from http://lsun.cs.princeton.edu

2. Prepare the dataset following the instructions below 

  1. Install LMDB in your system: 
   	`sudo apt-get install liblmdb-dev`
	
  2. Install torch packages:
   	```
	luarocks install lmdb.torch
	luarocks install tds
	```
	
  3. Once downloading bedroom_train_lmdb, unzip the dataset and put it in a directory `lsun/train`
   
  4. Create an index file :
	Copy lsun_index_generator.lua to lsun/train, and run
	```
	cd lsun/train
	DATA_ROOT=. th lsun_index_generator.lua
	```
	Now you should have bedroom_train_lmdb_hashes_chartensor.t7 in lsun/train
	
   5. Now return to the parent direcotry of lsun, and you should be ready to run lsgan.lua:
   	```
	DATA_ROOT=lsun th lsgan.lua
	```
	
### How to display the generated images
  
To display images during training and generation, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

## For Conditional LS-GAN (CLS-GAN)

1. Download and prepare datasets

  1.  MNIST:
         run `torch-rocks install https://raw.github.com/andresy/mnist/master/rocks/mnist-scm-1.rockspec`
  2. CIFAR10:
	 run `th ./data/Cifar10bintoTensor.lua`
  3. SVHN:
	 run `th ./data/svhn.lua`

2. Now you should be able to run clsgan.lua now. Select the dataset you want to use.  For example you want to run MNIST to generate handwritten digits according to ten digit classes. Then you can run the following command line

	```
	dataset=mnist th clsgan.lua
	```

   For the other parameters you can set, please refer to the script in clsgan.lua.



## Acknowledge: 

1. parts of codes are reused from DCGAN at https://github.com/Newmu/dcgan_code

2. the code downloading cifar10 is available at https://github.com/soumith/cifar.torch

3. the code downloading SVHN: http://ufldl.stanford.edu/housenumbers/ 




