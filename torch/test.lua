
require 'paths'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
local hdf5 = require('hdf5')
local matio = require('matio')
require('util')

torch.manualSeed(1)

opt_string = [[
    -h,--help                                       print help
    -g,--gpu_index          (default 0)             GPU index (start from 0)
    --model_path            (default "")            path to model
    --truncation            (default 3)             truncation in voxels
    --use_log_transform     (default 1)             use log transform
    --test_file             (default "")            path to file of scan to test
    --output_path           (default "output")      output path
    --visualize_matlab      (default 1)             use matlab to visualize the isosurface
    --classifier_path       (default "")            specify path to classifier net to use classification
]]

opt = lapp(opt_string)
-- print help or chosen options
if opt.help == true then
    print('Usage: th test.lua')
    print('Options:')
    print(opt_string)
    os.exit()
else
    print(opt)
end

cutorch.setDevice(opt.gpu_index + 1)

function loadh5(filename, truncation)
    local file = hdf5.open(filename, 'r');
    local input = file:read('data'):all()
    file:close()
    input = input:float()
    input[{ 1,{},{},{} }]:abs() --abs(sdf)
    if truncation then
        input[{ 1,{},{},{} }][input[{ 1,{},{},{} }]:gt(truncation)] = truncation
    end
    return input
end    

function saveisomeshes(dfs, filenames, isoval)
    assert(#dfs == #filenames, 'Must have a filename for every distance field')
    if isoval == nil then
        isoval = 1
    end
    local matfilenames = map(filenames, function(filename)
        return removeExtension(filename) .. '.mat'
    end)
    -- Generate .mat files and build up Matlab command
    local command = ''
    for i,objfilename in ipairs(filenames) do
        local df = dfs[i]
        local matfilename = matfilenames[i]
        matio.save(matfilename, df)
        command = command .. string.format("mat2obj('%s', '%s', %d);", matfilename, objfilename, isoval)
    end
    command = command .. 'exit;'
    -- Execute Matlab command
    os.execute(string.format('matlab -nodisplay -nosplash -nodesktop -r "%s"', command))
    -- Clean up .mat files
    for _,matfilename in ipairs(matfilenames) do
        os.execute(string.format('rm -f %s', matfilename))
    end
end

-- load net
assert(paths.filep(opt.model_path))
print('loading trained network from file: ' .. opt.model_path)
model = torch.load(opt.model_path)
cudnn.convert(model, cudnn)
model:evaluate()
--print(model)
local use_class = false
if not isempty(opt.classifier_path) then
    classifier = torch.load(opt.classifier_path):cuda()
    use_class = true
end

if paths.dirp(opt.output_path) == false then
    print('creating output prediction folder: ' .. opt.output_path)
    os.execute("mkdir " .. opt.output_path)
end


-- test function
function test(file, use_class)
    local inputVol = loadh5(file, opt.truncation)
    local inputSz = inputVol:size()
    local input = torch.Tensor(1, inputSz[1], inputSz[2], inputSz[3], inputSz[4]):cuda()
    input[1] = inputVol
    -- test sample
    local pred
    if use_class then
        local input_occ = input:clone()
        input_occ[{{},1,{},{},{}}] = torch.le(input[{{},1,{},{},{}}], 1):float() --already abs(sdf)
        input_occ[{{},2,{},{},{}}] = torch.eq(input[{{},2,{},{},{}}], 1):float()
        local classfts = classifier:forward(input_occ)
        classfts = nn.Unsqueeze(1):cuda():forward(classfts)
        pred = model:forward({input, classfts})
    else
        pred = model:forward(input)
    end
    local outfile = paths.concat(opt.output_path, paths.basename(file, '.h5') .. '.bin')
    if opt.use_log_transform then
        pred:exp()
        pred:add(-1)
    end
    pred = pred[1][1]:float()
    writeTensorToFile(pred, outfile)

    if opt.visualize_matlab then
        saveisomeshes({inputVol[1], pred}, {paths.concat(opt.output_path, 'input.obj'), paths.concat(opt.output_path, 'pred.obj')})
    end
end


test(opt.test_file, use_class)

