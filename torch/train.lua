
require 'paths'
require 'cunn'

require 'provider'

-- opts
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

torch.manualSeed(opt.manual_seed) 
if opt.gpu_index < 0 or opt.gpu_index > 1 then 
    print('error: gpu index out of bounds')
    error()
end
cutorch.setDevice(opt.gpu_index + 1) 

-- adam
solver_params = {
    learningRate = opt.learning_rate,
    beta1 = opt.beta1,
    beta2 = opt.beta2,
    epsilon = opt.epsilon
}

-- load data
local train_files = getLinesFromFile(opt.train_data)
local test_files  = getLinesFromFile(opt.test_data)
print('#train files = ' .. #train_files .. ', #test files = ' .. #test_files)

-- create model/criterion
paths.dofile(paths.concat('./models', opt.model .. '.lua'))

-- move to gpu
model = model:cuda()
criterion = criterion:cuda()
collectgarbage()

-- log results to files
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'))

parameters,gradParameters = model:getParameters()

function train(data_files)
    epoch = epoch or opt.start_epoch
 
    -- train!
    local train_file_indices = torch.randperm(#data_files)
    local train_error = 0
    for f = 1, #data_files do
        local idx = train_file_indices[f]
        assert(paths.filep(data_files[idx]))
        local dataset = loadDataBatch(data_files[idx], opt.max_jitter, opt.use_log_transform, opt.truncation)
        local err = train_batch(dataset)
        train_error = train_error + err
    end
    train_error = train_error / #data_files

    -- save/log current net
    if math.fmod(epoch + 1, opt.save_interval) == 0 then
        print('[ epoch ' .. epoch .. ', batchsize ' .. opt.batch_size .. ' ] train error = ' .. train_error)
        collectgarbage()
        local model_file = paths.concat(opt.save, 'model_' .. epoch .. '.net')
        model:clearState()
        torch.save(model_file, model) -- if gpu model, loads to gpu
    end
    epoch = epoch + 1

    return train_error
end

function train_batch(dataset)
    collectgarbage()

    --shuffle input
    local shuffle = torch.randperm(dataset:size()):cuda()
    
    local numInputChannels = dataset.data:size(2)
    local gridDim = dataset.data:size(3)
    local numTargetChannels = dataset.target:size(2) 
    
    local train_error = 0
    for t = 1,dataset:size(),opt.batch_size do
        -- create mini-batch
        local curBatchSize = math.min(opt.batch_size, dataset:size() - t + 1)
        if curBatchSize < opt.batch_size then break end -- make all batches same size
        local inputs = torch.Tensor(curBatchSize, numInputChannels, gridDim, gridDim, gridDim):cuda()
        local targets = torch.Tensor(curBatchSize, numTargetChannels, gridDim, gridDim, gridDim):cuda()
        for i = t, t+curBatchSize-1  do
            -- load new sample
            local sample = dataset[shuffle[i]]
            inputs[{i-t+1,{}}]  = sample[1]:cuda()
            targets[{i-t+1,{}}] = sample[2]:cuda()
        end
        local masks
        if opt.use_mask then
            masks = inputs[{{},{2},{},{},{}}]:clone()
            masks[masks:eq(1)] = 0 --mask out known
            masks[masks:eq(-1)] = 1
        end
        -- closure to evaluate f(X) and df/dX
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            --estimate f
            local output = model:forward(inputs)
            if opt.use_mask then
			    output:cmul(masks)
                targets:cmul(masks)
            end
            local f = criterion:forward(output, targets)
            --estimate df/dW
            local df_do = criterion:backward(output, targets)
            model:backward(inputs, df_do)

            train_error = train_error + f
            return f, gradParameters
        end 

        -- optimize on current mini-batch
        optim.adam(feval, parameters, solver_params) 
    end
    -- train error
    train_error = train_error / math.floor(dataset:size()/opt.batch_size)
    return train_error
end

-- test function
function test(data_files)
    collectgarbage()
    -- test!
    local test_error = 0
    for f = 1, #data_files do
        local dataset = loadDataBatch(data_files[f], opt.max_jitter, opt.use_log_transform, opt.truncation)
        local err = test_batch(dataset)
        test_error = test_error + err
    end
    test_error = test_error / #data_files
    return test_error
end

function test_batch(dataset)
    local numInputChannels = dataset.data:size(2)
    local gridDim = dataset.data:size(3)
    local numTargetChannels = dataset.target:size(2)

    local test_error = 0
    for t = 1,dataset:size(),opt.batch_size do
        local curBatchSize = math.min(opt.batch_size, dataset:size() - t + 1)
        if curBatchSize < opt.batch_size then break end -- make all batches same size
        local inputs = torch.Tensor(curBatchSize, numInputChannels, gridDim, gridDim, gridDim):cuda()
        local targets = torch.Tensor(curBatchSize, numTargetChannels, gridDim, gridDim, gridDim):cuda()
        for i = t, t+curBatchSize-1  do
            -- load new sample
            local sample = dataset[i]
            inputs[{i-t+1,{}}]  = sample[1]:cuda()
            targets[{i-t+1,{}}] = sample[2]:cuda()
        end
        local masks
        if opt.use_mask then
            masks = inputs[{{},{2},{},{},{}}]:clone()
            masks[masks:eq(1)] = 0 --mask out known
            masks[masks:eq(-1)] = 1
        end
        -- test sample
        local pred = model:forward(inputs)
        if opt.use_mask then
		    pred:cmul(masks)
            targets:cmul(masks)
        end
        local err = criterion:forward(pred, targets)
        test_error = test_error + err
    end
    -- testing error estimation
    test_error = test_error / math.floor(dataset:size()/opt.batch_size)
    return test_error
end


for i = 1, opt.max_epoch do
    local train_error = train(train_files)
    local test_error  = test(test_files)
 
    errLogger:add{['% train error']    = train_error, ['% test error']    = test_error}
    if opt.decay_learning_rate > 0 and epoch % opt.decay_learning_rate == 0 then
            solver_params.learningRate = solver_params.learningRate / 2
            print('{epoch ' .. epoch .. '} reduce learning rate to ' .. solver_params.learningRate)
    end
end
errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
errLogger:plot()

print('learning rate = ' .. opt.learning_rate)
-- re-test
local train_error = test(train_files)
local test_error  = test(test_files)
print('train err = ' .. train_error)
print('test  err = ' .. test_error)


local model_file = paths.concat(opt.save, 'model.net')
model:clearState()
torch.save(model_file, model)
print('saved model to ' .. model_file)


