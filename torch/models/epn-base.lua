
-- from model-orig, add more to fc and make another deconv layer

require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

-- create model
if (opt.retrain == 'none') then
    model = nn.Sequential()                                                    -- input 2 x 32^3
    -- conv part
    model:add(cudnn.VolumetricConvolution(2, 80, 4, 4, 4, 2, 2, 2, 1, 1, 1))  
    model:add(cudnn.VolumetricBatchNormalization(80))
    model:add(cudnn.ReLU())
    model:add(cudnn.VolumetricConvolution(80, 160, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    model:add(cudnn.VolumetricBatchNormalization(160))
    model:add(cudnn.ReLU())
    model:add(cudnn.VolumetricConvolution(160, 320, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    model:add(cudnn.VolumetricBatchNormalization(320))
    model:add(cudnn.ReLU())
    model:add(cudnn.VolumetricConvolution(320, 640, 4, 4, 4, 1, 1, 1, 0, 0, 0))
    model:add(cudnn.VolumetricBatchNormalization(640))
    model:add(nn.View(640))

    model:add(nn.Linear(640, 640))  
    model:add(cudnn.ReLU())
    model:add(nn.Linear(640, 640))  
    model:add(cudnn.ReLU())
    model:add(nn.View(640, 1, 1, 1))
    -- upconv part
    model:add(nn.VolumetricFullConvolution(640, 320, 4, 4, 4, 1, 1, 1, 0, 0, 0))
    model:add(cudnn.VolumetricBatchNormalization(320))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(320, 160, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    model:add(cudnn.VolumetricBatchNormalization(160))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(160, 80, 4, 4, 4, 2, 2, 2, 1, 1, 1))
    model:add(cudnn.VolumetricBatchNormalization(80))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(80, 1, 4, 4, 4, 2, 2, 2, 1, 1, 1))  

    -- re-weight to log space
    if opt.use_log_transform then
        model:add(nn.Abs())
        local addLayer = nn.Add(1, true) --always add 1 (since going to take ln)
        addLayer.bias = torch.ones(1)
        addLayer.accGradParameters = function() return end --fix the weights
        model:add(addLayer)
        model:add(nn.Log())
    end
else --preload network
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('loading previously trained network: ' .. opt.retrain)
    model = torch.load(opt.retrain)
end
cudnn.convert(model, cudnn)
print('model:')
print(model)

-- create criterion
criterion = nn.SmoothL1Criterion()
print('criterion:')
print(criterion)


