
-- from model-orig, add more to fc and make another deconv layer

require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
local nnutil = require('nnutil')

-- create model
if (opt.retrain == 'none') then
    --local nf = 64 --num feature maps
    --local nf = 32
    local nf = 80

    local input = {}
    local vol = nn.Identity()()
    table.insert(input, vol)

    -- conv part
    local enc1 = nnutil.VolConvBlock({ activation='LeakyReLU', doBatchNorm=false})(2, nf, 4, 2, 1)(vol)
    local enc2 = nnutil.VolConvBlock({ activation='LeakyReLU' })(nf, 2*nf, 4, 2, 1)(enc1)
    local enc3 = nnutil.VolConvBlock({ activation='LeakyReLU' })(2*nf, 4*nf, 4, 2, 1)(enc2)
    local enc4 = nnutil.VolConvBlock({ activation='LeakyReLU' })(4*nf, 8*nf, 4, 1, 0)(enc3)
    local encoded = enc4

    --model = nn.gModule(input, {encoded})
    local bottleneck = nn.Sequential()
    bottleneck:add(nn.View(8*nf))
    nnutil.FullyConnectedBlock({ doBatchNorm=false })(8*nf, 8*nf)(bottleneck)
    nnutil.FullyConnectedBlock({ doBatchNorm=false })(8*nf, 8*nf)(bottleneck)
    bottleneck:add(nn.View(8*nf, 1, 1, 1))
    local bottlenecked = bottleneck(encoded)
    --model = nn.gModule(input, {bottlenecked})

    --decoder
    local d1 = nn.JoinTable(2)({bottlenecked, enc4}) 
    local dec1 = nnutil.VolUpConvBlock()(2*8*nf, 4*nf, 4, 1, 0, 0)(d1)
    local d2 = nn.JoinTable(2)({dec1, enc3})
    local dec2 = nnutil.VolUpConvBlock()(2*4*nf, 2*nf, 4, 2, 1, 0)(d2)
    local d3 = nn.JoinTable(2)({dec2, enc2})
    local dec3 = nnutil.VolUpConvBlock()(2*2*nf, nf, 4, 2, 1, 0)(d3)
    local d4 = nn.JoinTable(2)({dec3, enc1})
    local decoded = nnutil.VolUpConvBlock({ activation='none', doBatchNorm=false })(2*nf, 1, 4, 2, 1, 0)(d4)

    -- re-weight to log space
    if opt.use_log_transform then
        decoded = nn.Log()(nn.AddConstant(1)(nn.Abs()(decoded)))
    end

    model = nn.gModule(input, {decoded})
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


