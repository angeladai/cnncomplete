#!/usr/bin/env th

require 'hdf5'

function jitter(src, tgt, jitter, truncation, tgtTruncation)
    local jitterList = torch.Tensor(src:size()[1], 3)
    local dst = torch.Tensor(src:size()):fill(truncation)
    dst[{{},2,{},{},{}}]:fill(1)
    local tgtdst
    if tgt then 
        tgtdst = torch.Tensor(tgt:size()):fill(tgtTruncation) 
    end
    for idx = 1,src:size()[1] do
        local i = math.random(-jitter, jitter)
        local j = math.random(-jitter, jitter)
        local k = math.random(-jitter, jitter)
        if i >= 0 then xidx = {i+1,dst:size(3),1,dst:size(3)-i} end
        if i < 0 then xidx = {1,dst:size(3)+i,-i+1,dst:size(3)} end
        if j >= 0 then yidx = {j+1,dst:size(4),1,dst:size(4)-j} end
        if j < 0 then yidx = {1,dst:size(4)+j,-j+1,dst:size(4)} end
        if k >= 0 then zidx = {k+1,dst:size(5),1,dst:size(5)-k} end
        if k < 0 then zidx = {1,dst:size(5)+k,-k+1,dst:size(5)} end
        --jitter both src and tgt
        dst[{{idx},{},{xidx[1],xidx[2]},{yidx[1],yidx[2]},{zidx[1],zidx[2]}}] = 
                src[{{idx},{},{xidx[3],xidx[4]},{yidx[3],yidx[4]},{zidx[3],zidx[4]}}]
        if tgt then tgtdst[{{idx},{},{xidx[1],xidx[2]},{yidx[1],yidx[2]},{zidx[1],zidx[2]}}] =
                tgt[{{idx},{},{xidx[3],xidx[4]},{yidx[3],yidx[4]},{zidx[3],zidx[4]}}] end
        jitterList[{{idx},{1}}] = i
        jitterList[{{idx},{2}}] = j
        jitterList[{{idx},{3}}] = k
    end
    return dst, tgtdst, jitterList
end

-- maxJitter: up to maxJitter movement in up/down/left/right
function getData(readFilename, maxJitter, useLogTransform, truncation)
    --print('loading dataset: ' .. readFilename)
    local readFile = hdf5.open(readFilename, 'r');
    local dataset = {
        data = readFile:read('data'):all(),
        target = readFile:read('target'):all()
    }
    readFile:close()
    -- size/indexing
    setmetatable(dataset, 
        {__index = function(t, i) 
                    return {t.data[i], t.target[i]} 
                end}
    );
    dataset.data = dataset.data:float()
    dataset.target = dataset.target:float()
    dataset.data[{ {},1,{},{},{} }]:abs() --abs(sdf)
    if truncation then
        --print('applying truncation of ' .. truncation)
        dataset.data[{ {},1,{},{},{} }][dataset.data[{ {},1,{},{},{} }]:gt(truncation)] = truncation
        dataset.target[{ {},{},{},{},{} }][dataset.target[{ {},{},{},{},{} }]:gt(truncation)] = truncation
    end
    local tgtTrunc = truncation
    if useLogTransform then -- since we will take log of model output, need to make sure target is in log space too
        --print('applying log transform to target')
        dataset.target = torch.add(dataset.target, 1)
        dataset.target[{ {},{},{},{},{} }]:log()
        tgtTrunc = torch.log(truncation + 1)
    end
    local jitterList = torch.Tensor(dataset.data:size()[1], 3):fill(0)
    if maxJitter > 0 then
        local numSamples = dataset.data:size()[1]
        for i = 1,numSamples,1000 do
            local beginidx = i
            local endidx = math.min(numSamples, i+1000)
            dataset.data[{ {beginidx, endidx}, {}, {}, {}, {} }], dataset.target[{ {beginidx, endidx}, {}, {}, {}, {} }], jitterList[{ {beginidx, endidx}, {} }] = 
                jitter(dataset.data[{ {beginidx, endidx}, {}, {}, {}, {} }], dataset.target[{ {beginidx, endidx}, {}, {}, {}, {} }], maxJitter, truncation, tgtTrunc)
        end
    end
    function dataset:size()
        return self.data:size(1)
    end

    return dataset, jitterList
end

-- when reading from file list
function loadDataBatch(filename, maxJitter, useLogTransform, truncation)
    local dataset = getData(filename, maxJitter, useLogTransform, truncation)
    return dataset
end

function getInputData(readFilename, maxJitter, useLogTransform, truncation)                                                                                 --print('loading dataset: ' .. readFilename)
    local readFile = hdf5.open(readFilename, 'r');                                                                                                     local dataset = {                                                                                                                                      data = readFile:read('data'):all(),
    }
    readFile:close()
    -- size/indexing
    setmetatable(dataset,
        {__index = function(t, i)
                    return {t.data[i]}                                                                                                                end}
    );
    dataset.data = dataset.data:float()
    dataset.data[{ {},1,{},{},{} }]:abs() --abs(sdf)
    if truncation then                                                                                                                                     print('applying truncation of ' .. truncation)                                                                                                     dataset.data[{ {},1,{},{},{} }][dataset.data[{ {},1,{},{},{} }]:gt(truncation)] = truncation
    end
    local jitterList = torch.Tensor(dataset.data:size()[1], 3):fill(0)
    if maxJitter > 0 then                                                                                                                                  local numSamples = dataset.data:size()[1]
        for i = 1,numSamples,1000 do
            local beginidx = i
            local endidx = math.min(numSamples, i+1000)
            local tmp
            dataset.data[{ {beginidx, endidx}, {}, {}, {}, {} }], tmp, jitterList[{ {beginidx, endidx}, {} }] =
                jitter(dataset.data[{ {beginidx, endidx}, {}, {}, {}, {} }], nil, maxJitter, truncation, nil)
        end
    end
    function dataset:size()
        return self.data:size(1)
    end

    return dataset, jitterList
end

