
-- from model-orig, add more to fc and make another deconv layer

require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

-- create model
if (opt.retrain == 'none') then
    model = nn.Sequential()                                                    -- input 2 x 32^3
    -- conv part
    model:add(cudnn.VolumetricConvolution(2, 32, 6, 6, 6, 2, 2, 2, 0, 0, 0))   -- output 32 x 14^3
    model:add(cudnn.VolumetricBatchNormalization(32))
    model:add(cudnn.ReLU())
    model:add(cudnn.VolumetricConvolution(32, 32, 3, 3, 3, 1, 1, 1, 0, 0, 0))  -- output 32 x 12^3
    model:add(cudnn.VolumetricBatchNormalization(32))
    model:add(cudnn.ReLU())
    model:add(cudnn.VolumetricConvolution(32, 32, 4, 4, 4, 2, 2, 2, 1, 1, 1))  -- output 32 x 6^3
    model:add(cudnn.VolumetricBatchNormalization(32))
    model:add(nn.View(6912))

    model:add(nn.Linear(6912, 512))    -- fully connected layer
    --model:add(nn.BatchNormalization(512))
    model:add(cudnn.ReLU())
    model:add(nn.Linear(512, 2048))
    --model:add(nn.BatchNormalization(2048))
    model:add(cudnn.ReLU())
    -- reshape
    model:add(nn.View(32, 4, 4, 4))  -- 2048 = 32*(4^3) 
    -- upconv part
    model:add(nn.VolumetricFullConvolution(32, 16, 4, 4, 4, 2, 2, 2, 1, 1, 1)) -- output 16 x 8^3
    model:add(cudnn.VolumetricBatchNormalization(16))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(16, 8, 4, 4, 4, 2, 2, 2, 1, 1, 1))  -- output 8 x 16^3
    model:add(cudnn.VolumetricBatchNormalization(8))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(8, 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))   -- output 4 x 32^3
    model:add(cudnn.VolumetricBatchNormalization(4))
    model:add(cudnn.ReLU())
    model:add(nn.VolumetricFullConvolution(4, 1, 5, 5, 5, 1, 1, 1, 2, 2, 2))   -- output 1 x 32^3

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


