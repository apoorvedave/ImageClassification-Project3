require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'cunn'
require 'cutorch'
torch.setdefaulttensortype("torch.FloatTensor")

coefL1 = 0
coefL2 = 0
batchSize = 10
run='SGD'

opt = {save = run, optimization = run, learningRate =  1e-3,
 weightDecay = 0,
 momentum = 0,
 learningRateDecay = 5e-7, batchsize=10, batchSize=10, threads=8, epochs = 30}

-- ('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
-- ('-network', '', 'reload pretrained network')
-- ('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
-- ('-full', false, 'use full dataset (50,000 samples)')
-- ('-visualize', false, 'visualize input data and weights during training')
-- ('-seed', 1, 'fixed input seed for repeatable experiments')
-- ('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
-- ('-learningRate', 1e-3, 'learning rate at t=0')
-- ('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
-- ('-weightDecay', 0, 'weight decay (SGD only)')
-- ('-momentum', 0, 'momentum (SGD only)')
-- ('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
-- ('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
-- ('-threads', 2, 'nb of threads to use')

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(3,3,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.

model:add(nn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(3,3,2,2))

model:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(3,3,2,2))

model:add(nn.SpatialAveragePooling(3,3))
model:add(nn.View(64))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model:add(nn.Linear(64, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()
collectgarbage()

trsize = 50000
tesize = 10000

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('../cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('../cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

require 'cunn';
require 'cutorch'
trainData.data = trainData.data:cuda()
trainData.labels = trainData.labels:cuda()
testData.data = testData.data:cuda()
testData.labels = testData.labels:cuda()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()


----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter % 10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{
            image=model:get(1).weight, zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='stage 1: weights', padding=1
         }
         win_w2 = image.display{
            image=model:get(4).weight, zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='stage 2: weights', padding=1
         }
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{
            image=W1, zoom=0.5, min=-1, max=1,
            win=win_w1, legend='W1 weights'
         }
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{
            image=W2, zoom=0.5, min=-1, max=1,
            win=win_w2, legend='W2 weights'
         }
      end
   end
   iter = iter + 1
end

function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   --print('<trainer> online epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = dataset.data[{ {t,math.min(t+opt.batchSize-1,dataset:size())} }]
      local targets = dataset.labels[{ {t,math.min(t+opt.batchSize-1,dataset:size())} }]

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if coefL1 ~= 0 or coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + coefL1 * norm(parameters,1)
            f = f + coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(coefL1) + parameters:clone():mul(coefL2) )
         end

         -- update confusion
         for i = 1,batchSize do
            confusion:add(outputs[i], targets[i])
         end
            
         trainError = trainError + f

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/10)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'cifar.net')
   -- if paths.filep(filename) then
   --   os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   -- end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)
   -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   --print('<trainer> on testing Set:')
   --local inputs = dataset.data
   --local targets = dataset.labels
   --local preds = model:forward(inputs)
   --for i = 1,dataset:size() do
   --   confusion:add(preds[i], targets[i])
   --   err = criterion:forward(preds[i], targets[i])
   --   testError = testError + err
   --end
    
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = dataset.data[{ {t,math.min(t+opt.batchSize-1,dataset:size())} }]
      local targets = dataset.labels[{ {t,math.min(t+opt.batchSize-1,dataset:size())} }]
      local preds = model:forward(inputs)
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
         err = criterion:forward(preds[i], targets[i])
         testError = testError + err
      end
   end
   
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   --print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   print("Dave testAccuracy: " .. testAccuracy)
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end

----------------------------------------------------------------------
-- and train!
--
for i = 1,opt.epochs do
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)

   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   --accLogger:plot()
   --errLogger:plot()
end
