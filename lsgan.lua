--[[
Loss-Sensitive GAN

File: lsgan.lua
Author: Guo-Jun Qi, guojunq@gmail.com
Date: 2/7/2017

This implements LS-GAN with NO condition as input. It shows the LS-GAN model.  For conditional LS-GAN for the classification tasks, please refer to clsgan.lua.
--]]

require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,--64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0001,            -- initial learning rate for adam
   beta1 = 0.5,--0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 12,        -- display window id.
   gpu = 1,                -- gpu = -1 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'lsgan_result',
   noise = 'uniform',       -- uniform / normal
   lambda=0.0002,              -- the scale of the distance metric used for adaptive margins. This is actually tau in the original paper. L2: 0.05/L1: 0.001, temporary best 0.008 before applying scaling, 
   gamma = 0.,		      -- the coefficient for loss minimization term.  Set to zero for non-conditional LS-GAN as the theorem shows this term can be ignored. But setting it to a small value allows a more stable result.
   decay_rate = 0.00,  -- weight decay: 0.00005 
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
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
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

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = -1 -- the original one was 1 , we changed that for sake of MarginCriterion
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))--
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))--
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))--
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
--netD:add(nn.Sigmoid())
----------comment out-----------------------
--netD:add(nn.LogSigmoid())--original Sigmoid
--netD:add(nn.MulConstant(-1,false))
--------------------------------------------
-- state size: 1 x 1 x 1

--netD:add(nn.SoftPlus())
netD:add(nn.ReLU(true))
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

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
   learningRate = 0.0001,--0.004,
   learningRateDecay=1.000004,
   momentum = 0.,--opt.beta1,
}
optimStateDsgd = {
   learningRate = 0.0001,--0.008,
   learningRateDecay=1.000004,
   momentum = 0.,--opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local input_fakeimg=torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local df_mnllik=(1/(opt.batchSize))*torch.ones(opt.batchSize,1) -- changed by GQ
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

----------------------------------------------------------------------------
if opt.gpu > -1  then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda();    input_fakeimg=input_fakeimg:cuda(); df_mnllik=df_mnllik:cuda()

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

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local outputR = netD:forward(input):clone()

   -- term 1 of cost negetive log liklihood, I disable this part now.
   local mnllik=torch.mean(opt.gamma*outputR) -- changed by GQ: remove the factor of -1
   netD:backward(input,opt.gamma*df_mnllik)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input_fakeimg:copy(fake)
   local pdist=L1dist:forward({input:view(opt.batchSize,3* opt.fineSize* opt.fineSize),input_fakeimg:view(opt.batchSize,3* opt.fineSize* opt.fineSize)})
   pdist:mul(opt.lambda) -- for discriminator this will beome constant doesn't need backward

   --pdist:fill(0.2)
   local outputF = netD:forward(input_fakeimg):clone()
   local cost1=pdist+outputR-outputF -- changed by GQ: mul(-1) moved to outputF
   costR = outputR:mean()
   costF = outputF:mean()
   mar = pdist:mean()
  
   local error_hinge = criterion:forward(cost1, label)
   local df_error_hinge = criterion:backward(cost1, label)

   netD:backward(input_fakeimg, -1*df_error_hinge) -- changed by GQ: add mul(-1)

   accGradD = gradParametersD:clone()

   gradParametersD:zero()
   netD:forward(input) -- we have to run the forward pass one more time on input of real image to make sure that the backward gradients are computed correctly on netD
   netD:backward(input,df_error_hinge) -- changed by GQ: add mul(-1) to change it back
   accGradD = accGradD + gradParametersD
   errD = error_hinge + mnllik

   return errD, accGradD+opt.decay_rate*x
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   gradParametersD:zero()

   local outputF = netD:forward(input_fakeimg)
   errG = torch.mean(outputF)
   local df_error_hinge=(1/(opt.batchSize))*outputF:clone():fill(1)
   local df_outputF = netD:updateGradInput(input_fakeimg,df_error_hinge)
   netG:backward(noise,df_outputF)

   return errG, gradParametersG+opt.decay_rate*x
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()


        -- (1) Update loss function network:
      optim.adam(fDx, parametersD, optimStateD)-- original
        --optim.sgd(fDx, parametersD, optimStateDsgd)

        -- (2) Update G network: 
      optim.adam(fGx, parametersG, optimStateG)
        --optim.sgd(fGx, parametersG, optimStateGsgd)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then -- original counter % 10
          --if opt.noise == 'uniform' then -- regenerate random noise
          --   noise_vis:uniform(-1, 1)
          --elseif opt.noise == 'normal' then
          --   noise_vis:normal(0, 1)
          --end
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f   costR:%.4f   costF:%.4f   meanD:%.4f   gradD:%.4f   gradG:%.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1, costR, costF, mar, torch.mean(torch.abs(accGradD)), torch.mean(torch.abs(gradParametersG))))
      end
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
end
