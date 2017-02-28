
-- from model-orig, add more to fc and make another deconv layer

require 'nn'
require 'cunn'
require 'cudnn'
require('nngraph')

local function mergeDefaults(tbl, defaultTbl)
    if defaultTbl then
        local newTbl = {}
        for k,v in pairs(tbl) do newTbl[k] = v end
        for k,v in pairs(defaultTbl) do
            if newTbl[k] == nil then newTbl[k] = v end
        end
        return newTbl
    else
        return tbl
    end
end

local function checkOpts(opts)
    opts = opts or {}
    opts = mergeDefaults(opts, {
        activation = 'ReLU',
        leakyReluSlope = 0.2,
        doBatchNorm = true,
        batchNormEps = 1e-3
    })
    return opts
end

local function addModulesToSeq(nnseq, modules)
    for _,mod in ipairs(modules) do
        nnseq:add(mod)
    end
    return nil
end

local function addModulesToGraph(node, modules)
    for _,mod in ipairs(modules) do
        node = mod(node)
    end
    return node
end

local SequentialMT = getmetatable(nn.Sequential)
local NodeMT = getmetatable(nngraph.Node)
local function addModules(x, modules)
    local mt = getmetatable(x)
    if mt == SequentialMT then
        return addModulesToSeq(x, modules)
    elseif mt == NodeMT then
        return addModulesToGraph(x, modules)
    else
        print(mt)
        error('addModules only accepts nn.Sequential or nngraph.Node inputs')
    end
end

local function activationModules(modules, opts)
    if opts.activation == 'ReLU' then
        table.insert(modules, cudnn.ReLU(true))
        print('cudnn.ReLU')
    elseif opts.activation == 'LeakyReLU' then
        table.insert(modules, nn.LeakyReLU(opts.leakyReluSlope, true):cuda())
        print('nn.LeakyReLU')
    elseif opts.activation == 'none' then
        print('no activation')
    else
        error('Unrecognized activation ' .. opts.activation)
    end
end

local function VolConvBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut, size, stride, pad)
        return function(x)
            local modules = {cudnn.VolumetricConvolution(nIn, nOut, size, size, size, stride, stride, stride, pad, pad, pad)}
            print('cudnn.VolumetricConvolution')
            if opts.doBatchNorm then
                table.insert(modules, cudnn.VolumetricBatchNormalization(nOut, opts.batchNormEps))
                print('cudnn.VolumetricBatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

local function VolUpConvBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut, size, stride, pad, extra)
        return function(x)
            --local modules = {cudnn.VolumetricFullConvolution(nIn, nOut, size, size, size, stride, stride, stride, pad, pad, pad, extra, extra, extra)}
            local modules = {nn.VolumetricFullConvolution(nIn, nOut, size, size, size, stride, stride, stride, pad, pad, pad)}
            print('nn.VolumetricFullConvolution')
            if opts.doBatchNorm then
                table.insert(modules, cudnn.VolumetricBatchNormalization(nOut, opts.batchNormEps))
                print('cudnn.VolumetricBatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

local function FullyConnectedBlock(opts)
    opts = checkOpts(opts)
    return function(nIn, nOut)
        return function(x)
            local modules = {nn.Linear(nIn, nOut)}
            print('nn.Linear')
            if opts.doBatchNorm then
                table.insert(modules, cudnn.BatchNormalization(nOut, opts.batchNormEps))
                print('cudnn.BatchNormalization')
            end
            activationModules(modules, opts)
            return addModules(x, modules)
        end
    end
end

return {
    VolConvBlock = VolConvBlock,
    VolUpConvBlock = VolUpConvBlock,
    FullyConnectedBlock = FullyConnectedBlock,
}

