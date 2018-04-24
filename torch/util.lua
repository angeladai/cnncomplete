#!/usr/bin/env th

function jitterTensor(tensor, jitter, truncation, jitterMultFactor)
    jitterMultFactor = jitterMultFactor or 1
    local dst = torch.Tensor(tensor:size()):fill(truncation)
    if dst:size(1) == 2 then dst[{2,{},{},{}}]:fill(1) end
    local i = jitterMultFactor * jitter[1]
    local j = jitterMultFactor * jitter[2]
    local k = jitterMultFactor * jitter[3]
    if i >= 0 then xidx = {i+1,tensor:size(2),1,tensor:size(2)-i} end
    if i < 0 then xidx = {1,tensor:size(2)+i,-i+1,tensor:size(2)} end
    if j >= 0 then yidx = {j+1,tensor:size(3),1,tensor:size(3)-j} end
    if j < 0 then yidx = {1,tensor:size(3)+j,-j+1,tensor:size(3)} end
    if k >= 0 then zidx = {k+1,tensor:size(4),1,tensor:size(4)-k} end
    if k < 0 then zidx = {1,tensor:size(4)+k,-k+1,tensor:size(4)} end
    --perform the jitter
    dst[{{},{xidx[1],xidx[2]},{yidx[1],yidx[2]},{zidx[1],zidx[2]}}] =
            tensor[{{},{xidx[3],xidx[4]},{yidx[3],yidx[4]},{zidx[3],zidx[4]}}]
    return dst
end

function trim(s)
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- get all lines from a file
function getLinesFromFile(file)
  --print('[getlinesfromfile] ' .. file)
  assert(paths.filep(file))
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = trim(line)
  end
  return lines
end

function writeTensorToFile(tensor, filename)
    if not tensor then return false end
    local file = torch.DiskFile(filename, "w"):binary()
    local n = tensor:nElement()
    local s = tensor:float():storage()
--[[    local sz = tensor:size()
    file:writeInt(sz:size())
    for i = 1,sz:size() do
        file:writeInt(sz[i])
    end--]]
    return assert(file:writeFloat(s) == n)
end

function readFilesAndLabels( filename )
    assert(paths.filep(filename))
    filenames = {}
    labels = {}
    local file = io.open(filename)
    if file then
        for line in file:lines() do
            local name, label = unpack(line:split(" "))
            --name = paths.basename(name, dataType)
            nameParts = name:split("/")
            name = nameParts[#nameParts-1] .. '/' .. nameParts[#nameParts]
            table.insert(filenames, name)
            table.insert(labels, label)
        end
        file:close()
    else
        print('unable to open file: ' .. filename)
    end
    return filenames, labels
end


function splitstr(str, sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    str:gsub(pattern, function(c) fields[#fields+1] = c end)
    return fields
end

function removeExtension(str)
    return splitstr(str, '.')[1]
end

function map(tbl, fn)
    local ret = {}
    for k,v in pairs(tbl) do
        ret[k] = fn(v, k)
    end
    return ret
end

function isempty(s)
    return s == nil or s == ''
end