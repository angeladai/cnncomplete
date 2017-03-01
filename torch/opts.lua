--for cmd line arguments

require 'util'

local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Options:')
    -------------------- General options --------------------
    cmd:option('-save',          './logs/', 'subdirectory in which to save experiments')
    cmd:option('-train_data',    'train_shape_voxel_data_list.txt', 'txt list of h5 train files')
    cmd:option('-test_data',     'test_shape_voxel_data_list.txt', 'txt list of h5 test files')
    cmd:option('-truncation',    3,       'sdf truncation value (in voxels)')
    cmd:option('-use_log_transform', true, 'log transform sdf values')
    cmd:option('-use_mask',          false,'mask out known values')
    cmd:option('-gpu_index',     0,       'default gpu')
    cmd:option('-save_interval', 20,      'interval to save current state of model')
    cmd:option('-manual_seed',   1,       'manual seed')
    -------------------- Training options --------------------
    cmd:option('-max_epoch',    150,     '#epochs to run')
    cmd:option('-batch_size',    64,      'mini-batch size')
    cmd:option('-start_epoch',   1,       'manual epoch number (for restarts)')
    cmd:option('-max_jitter',    2,       'amount to translationally jitter, no data dup (0 for none)')
    -------------------- Optimization options --------------------
    cmd:option('-learning_rate', 0.001,   'learning rate')
    cmd:option('-decay_learning_rate', 50,'decay learning rate by half every n epochs')
    -- adam
    cmd:option('-beta1', 	     0.9,	  'first moment coefficient')
    cmd:option('-beta2',         0.999,   'second moment coefficient')
    cmd:option('-epsilon',       1e-8,    'for numerical stability')
    -------------------- Model options --------------------
    cmd:option('-model',       'epn-unet-class', 'model type')
    cmd:option('-trained_class_model', 'models/classifier-partial.net', 'classification model')
    cmd:option('-retrain',     'none',      'provide path to model to retrain with')


    local opt = cmd:parse(arg or {})
    if paths.dirp(opt.save) == false then
        print('creating log folder: ' .. opt.save)
        os.execute("mkdir " .. opt.save)
    else
        io.stderr:write(sys.COLORS.red .. 'warning: save folder already exists!' .. sys.COLORS.none)
        io.stderr:write(' (press key to continue)')
        io.read()
    end

    -- save to log
    cmd:log(paths.concat(opt.save, 'log.txt'), opt)
    cmd:silent()
    return opt
end


return M

