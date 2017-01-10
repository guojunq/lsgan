--[[
clsgan.lua: Conditional Loss-Sensitive GAN 

Author: Guo-Jun Qi, Date:1/9/2017

This implements the CLSGAN that takes an input of class label and outputs images corresponding to given labels.
We use three datasets available to train the model, and the users can provide their own by modifying the dataset loading procedure.

--]]



--require('mobdebug').start() -- for debug purpose

require 'torch'
require 'nn'
require 'optim'
require 'image'


opt = {
   dataset = 'svhn',       -- svhn / cifar10 / mnist: now we support these three datasets. Users should modify the loading of dataset and get_minibatch function to use their own datasets.
   batchSize = 64,--64,
   nex = 10,                -- #  of examples to produce for each class
   loadSize = 96,
   fineSize = 32,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   nlabel = 10,            -- #  of labels to be used
   nchannel = 3,           -- #  of input image channels
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.001,--0.000002, 0.0001           -- initial learning rate for adam
   beta1 = 0.5,--0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 25,        -- display window id.
   gpu = 1,                -- gpu = -1 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'svhn_32x32',
   noise = 'normal',       -- uniform / normal
   lambda=0.0008,            -- L2: 0.05/L1: 0.001
   gamma = 1.0, --0.00005,
   gamma_decay = 0.999,
   decay_rate = 0.0,  -- last: 0.02
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
--local DataLoader = paths.dofile('data/data.lua')
--local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
--print("Dataset: " .. opt.dataset, " Size: ", data:size())


----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = opt.nchannel
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = -1 -- the original one was 1 , we changed that for sake of MarginCriterion
local fake_label = 0
local nlabel = opt.nlabel

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz+nlabel, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16

------------------------ output 32x32 ---------------------------
netG:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
-- state size: (nc) x 32 x 32
------------------------------------------------------------------

--netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
--netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32

--netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

local netD = nn.Sequential()


-- input is (nc) x 32 x 32 
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 16 x 16 or 32 x 32
netD:add(SpatialConvolution(ndf, ndf, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 16 x 16 or 32 x 32

netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 8 x 8 or 16 x 16
netD:add(SpatialConvolution(ndf*2, ndf*2, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 8 x 8 or 16 x 16

netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 4 x 4 or 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 4, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 4 x 4 or 8 x 8

---------------- for input 64 x 64 ----------------------------
--netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
--netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
--netD:add(SpatialConvolution(ndf * 8, nlabel, 4, 4))
-- state size: nlabel x 1 x 1

-----------------for input 32 x 32 ------------------------------
netD:add(SpatialConvolution(ndf * 4, nlabel, 4, 4))
-- state size: nlabel x 1 x 1
-----------------------------------------------------------------
netD:add(nn.SpatialLogSoftMax())
netD:add(nn.MulConstant(-1,true)) -- because this is loss function and we will minimize it

--netD:add(nn.Sigmoid())
----------comment out-----------------------
--netD:add(nn.LogSigmoid())--original Sigmoid
--netD:add(nn.MulConstant(-1,false))
--------------------------------------------
-- state size: 1 x 1 x 1

--netD:add(nn.SoftPlus())
--netD:add(nn.ReLU(true))
netD:add(nn.View(nlabel):setNumInputDims(3))
-- state size: nlabel

netD:apply(weights_init)

--local criterion = nn.BCECriterion()
local criterion = nn.MarginCriterion(0) --set the coresponding y to -1 so it will become loss(x,y)=sum_i(max(0,0-(-1)*x[i]))/x:nElement()
--local criterion = nn.SoftMarginCriterion()

local L2dist=nn.PairwiseDistance(2)
local L1dist=nn.PairwiseDistance(1)
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}


optimStateGsgd = {
   learningRate = 0.004,
   learningRateDecay=1.000004,
   momentum = 0.5,--opt.beta1,
}
optimStateDsgd = {
   learningRate =0.008,
   learningRateDecay=1.000004,
   momentum = 0.5,--opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_label = torch.Tensor(opt.batchSize)
local input_label_onehot = torch.Tensor(opt.batchSize, opt.nlabel)
local noise = torch.Tensor(opt.batchSize, nz+nlabel, 1, 1)
local input_fakeimg=torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local df_mnllik=(1/(opt.batchSize))*torch.ones(opt.batchSize,1) -- changed by GQ
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

noise_vis = torch.Tensor(opt.nex*nlabel,nz+nlabel,1,1)               --noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

for i=1,opt.nex do
  for j=1,nlabel do
    noise_vis[{{(i-1)*nlabel+j},{1,nlabel},{1},{1}}]:fill(0)
    noise_vis[{{(i-1)*nlabel+j},{j},{1},{1}}]=1
  end
end

--print(noise_vis:size())

------------ function to load datasets--------------------------------------
print('beginning to load dataset...')
dataset = opt.dataset
if dataset == 'mnist' then
   mnist=require 'mnist'
   local trainset=mnist.traindataset()
   local testset=mnist.testdataset()
   c10= {
	data=image.scale(trainset.data,opt.fineSize,opt.fineSize):view(60000,1,opt.fineSize,opt.fineSize),
	label=trainset.label
	}
   c10t= {
	data=image.scale(testset.data,opt.fineSize,opt.fineSize):view(10000,1,opt.fineSize,opt.fineSize),
	label=testset.label
	}
   size_train=trainset.size
   size_test=testset.size
   print('loading mnist')
elseif dataset == 'cifar10' then
   c10=torch.load('cifar10-train.t7')
   size_train=c10.label:size()[1]
   c10t=torch.load('cifar10-test.t7')
   size_test=c10t.label:size()[1]
   print('loading cifar10')
elseif dataset == 'svhn' then
   local loaded=torch.load('./housenumbers/train_32x32.t7','ascii')
   c10= {
      data=loaded.X:transpose(3,4),
      label=loaded.y[1]-1
   }
   size_train=c10.label:size()[1]
   local loaded_t=torch.load('./housenumbers/test_32x32.t7','ascii')
   c10t = {
      data=loaded_t.X:transpose(3,4),
      label=loaded_t.y[1]-1
   }
   size_test=c10t.label:size()[1]
   print('loading svhn')
else
   print('no data is loading...')
end

------------------ we should scale the image pixel values to [-1,1] in line with tanh output by the generator --------------------
if dataset == 'svhn' or dataset=='cifar10' or dataset == 'mnist' then
   c10.data=c10.data:double()
   c10.data=2*(c10.data/255)-1
   c10t.data=c10t.data:double()
   c10t.data=2*(c10t.data/255)-1
end
----------------- end of scaling --------------------------------------------------------------------------------------------------

print('finishing loading dataset...')

local start_idx=1
local end_idx=start_idx+opt.batchSize-1

local function mnist_getBatch()
   local i=1
   input_label_onehot:fill(0)

   for j=start_idx,end_idx do
      -- print(input[{{i},{},{},{}}]:size())
       
       if dataset == 'other' then
         input[{{i},{},{},{}}][1]=2*(image.scale(trainset[j].x,opt.fineSize,opt.fineSize)/255)-1
         --input[{{i},{},{},{}}]=trainset[j].x
         input_label[i]=trainset[j].y+1
         input_label_onehot[i][trainset[j].y+1]=1
       elseif dataset == 'cifar10' or dataset == 'svhn' or dataset == 'mnist' then
         --input[{{i},{},{},{}}]=image.scale(c10.data[j],opt.fineSize,opt.fineSize)
         input[{{i},{},{},{}}]:copy(c10.data[j]) 
         input_label[i]=c10.label[j]+1
         input_label_onehot[i][c10.label[j]+1]=1
       end 
       i=i+1
   end
   start_idx=(start_idx+opt.batchSize-1)%size_train+1
   end_idx = (start_idx+opt.batchSize-2)%size_train+1
end

----------------------------------------------------------------------------
if opt.gpu > -1  then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda();    input_fakeimg=input_fakeimg:cuda(); df_mnllik=df_mnllik:cuda();
   input_label_onehot = input_label_onehot:cuda(); noise_vis=noise_vis:cuda(); 

   if dataset == 'cifar10' or dataset == 'svhn' or dataset == 'mnist' then
      c10t.data = c10t.data:double():cuda()
   end

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
      cudnn.convert(L2dist, cudnn)
      cudnn.convert(L1dist, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda();     L2dist:cuda();     L1dist:cuda();
end

parametersD, gradParametersD = netD:getParameters()
parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end



--local nllik=nn.Sequential()
--nllik.add(nn.Mul)

 -- make the pairwise distance between real image and fake image global to save computation! ok we can not save computation :P

local function set_noise(n,l)
      for i = 1, opt.batchSize do
         n[{{i},{1,nlabel},{1},{1}}]:fill(0)
         n[i][l[i]][1][1]=1
      end
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   --local real = data:getBatch()
   mnist_getBatch()
   data_tm:stop()
   --input:copy(real)

   
   label:fill(real_label)

   local outputR = netD:forward(input):clone()


   -- term 1 of cost negetive log liklihood, I disable this part now.
   mnllik=torch.mean(torch.sum(opt.gamma*torch.cmul(outputR,input_label_onehot),2)) -- changed by GQ: remove the factor of -1

   
   netD:backward(input, opt.gamma*input_label_onehot/opt.batchSize) -- df_mnllik = opt.gamma*input_label_onehot/opt.batchSize

   --opt.gamma = opt.gamma * opt.gamma_decay

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end

   set_noise(noise,input_label) -- set_noise will attach one_hot encoding of labels to noise to output conditioned samples.

   local fake = netG:forward(noise)
   input_fakeimg:copy(fake)
   local pdist=L1dist:forward({input:view(opt.batchSize,nc* opt.fineSize* opt.fineSize),input_fakeimg:view(opt.batchSize,nc* opt.fineSize* opt.fineSize)})

   pdist:mul(opt.lambda) -- for discriminator this will beome constant doesn't need backward

   
   local outputF = netD:forward(input_fakeimg):clone()
   
   cost1R = torch.sum(torch.cmul(outputR,input_label_onehot),2)
   cost1F = torch.sum(torch.cmul(outputF,input_label_onehot),2)
   local cost1=pdist+cost1R-cost1F
   mar = pdist:mean()
  
   local error_hinge = criterion:forward(cost1, label)
   local df_error_hinge = criterion:backward(cost1, label)



   netD:backward(input_fakeimg, -1*torch.cmul(df_error_hinge:view(opt.batchSize,1):expandAs(input_label_onehot),input_label_onehot)) -- changed by GQ: add mul(-1)

   --accGradD = gradParametersD:clone()


   --gradParametersD:zero()
   netD:forward(input) -- we have to run the forward pass one more time on input of real image to make sure that the backward gradients are computed correctly on netD

   netD:backward(input,torch.cmul(df_error_hinge:view(opt.batchSize,1):expandAs(input_label_onehot),input_label_onehot)) -- changed by GQ: add mul(-1) to change it back

   
   --accGradD = accGradD + gradParametersD
   

   errD = error_hinge  + mnllik

   --print(('gradD:%.4f'):format(torch.mean(torch.abs(accGradD))))
   --print(('gradD:%.4f'):format(torch.mean(torch.abs(gradParametersD))))




   return errD, gradParametersD+opt.decay_rate*x --accGradD+opt.decay_rate*x
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   gradParametersD:zero()

   --local fake_img = netG:forward(noise)
     
   local outputF = netD:forward(input_fakeimg)
   
   errG = torch.mean(torch.cmul(outputF,input_label_onehot)) * nlabel
   
   local df_error_hinge=(1/(opt.batchSize))* input_label_onehot -- outputF:clone():fill(1)
   --pow(outputF,0)

   local df_outputF = netD:updateGradInput(input_fakeimg,df_error_hinge)
   --local df_outputF = netD:backward(fake_img,df_error_hinge)
   netG:backward(noise,df_outputF)

   --print(('gradG:%.4f'):format(torch.mean(torch.abs(gradParametersG))))

   return errG, gradParametersG+opt.decay_rate*x
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   counter = 0
   for i = 1, math.min(size_train, opt.ntrain), opt.batchSize do
      tm:reset()

      --counterN=torch.round(counter/1)

      --if counterN%2==0 then
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)-- original
        --optim.sgd(fDx, parametersD, optimStateDsgd)
      print(('mean of parametersD:%.10f'):format(parametersD*parametersD))
      --end

      --if counterN%2 == 1 then
        -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)
        --optim.sgd(fGx, parametersG, optimStateGsgd)
      print(('mean of parametersG:%.10f'):format(parametersG*parametersG))
      --end

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then -- original counter % 10

          local fake = netG:forward(noise_vis)
          --local real = data:getBatch()
          real=input:clone()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name, nrow=opt.nex})
      end



      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f   mnllik: %.4f    costR:%.4f   costF:%.4f   meanD:%.4f  mem:%d kb'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(size_train, opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1, mnllik, cost1R:mean(), cost1F:mean(), mar, collectgarbage("count")))
      end
      collectgarbage()
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))


----------------------------------- test-------------------------------------------------------

   if dataset == 'svhn' or dataset == 'cifar10' or dataset == 'mnist' then
      local outputT = netD:forward(c10t.data)
      
      local mpred, pred = torch.min(outputT,2)
      local err = torch.ne(pred:view(size_test):long(),(c10t.label+1):long()):sum()
      print(('Error number: %d, Test error: %.4f'):format(err, err/size_test))
   end
end


   