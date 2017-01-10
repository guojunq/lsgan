Questions about the source codes can be directed to Dr. Guo-Jun Qi at guojunq@gmail.com.  All copyrights reserved.

Author: Guo-Jun Qi, Date: 1/9/2017

1. Download and prepare datasets

   1.1. MNIST
         run "torch-rocks install https://raw.github.com/andresy/mnist/master/rocks/mnist-scm-1.rockspec"
   1.2. CIFAR10
	 run "th ./data/Cifar10bintoTensor.lua" 
   1.3. SVHN
	 run "th ./data/svhn.lua"

2. Now you should be able to run clsgan.lua now. Select the dataset you want to use.  For example you want to run MNIST to generate handwritten digits according to ten digit classes. Then you can run the following command line

	dataset=mnist th clsgan.lua

   For the other parameters you can set, please refer to the script in clsgan.lua.

3. To display images during training and generation, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)


Acknowledge: 

1. parts of codes are reused from DCGAN at 

https://github.com/Newmu/dcgan_code

2. the code downloading cifar10 is available at https://github.com/soumith/cifar.torch

3. the code downloading SVHN: http://ufldl.stanford.edu/housenumbers/ 




