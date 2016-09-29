function recursivePut2Gpu(Atable, AgpuTable)
    for i=1,#Atable do
        if torch.type(Atable[i]) == 'table' then
            if AgpuTable[i] == nil then
                AgpuTable[i] = {}
            end
            recursivePut2Gpu(Atable[i], AgpuTable[i])
        else
            if AgpuTable[i] == nil then
                AgpuTable[i] = torch.CudaTensor()
            end
            AgpuTable[i]:resize(Atable[i]:size()):copy(Atable[i])
        end
    end
end
function put2GPU(cpuData, gpuLocation)
    local dataInd = 0
    if torch.type(gpuLocation) == 'table' then
        recursivePut2Gpu(cpuData, gpuLocation)
    else
        if #cpuData == 1 then
            gpuLocation:resize(cpuData[1]:size()):copy(cpuData[1])
        else
            error('Some kind of error...')
        end
    end
end

function newInfoEntry(vname, vval, vn, store_all)
    local storeAll = store_all or false -- if true, vn will be ignored
    return {name=vname, value=vval, N=vn, store=storeAll}
end

function topK(prediction, target, K)
    local acc = 0
    local _,prediction_sorted = prediction:sort(2, true) -- descending
    local batch_Size = prediction:size(1)
    for i=1,batch_Size do
        for j=1,K do
            if prediction_sorted[i][j] == target[i] then
                acc = acc + 1
                break
            end
        end
    end
    return acc/batch_Size
end

function meanAvgPrec(prediction, target, dim, threshold)
    -- compute mean Average Precision for input minibatch
    -- Args:
    --   prediction: batchSize * nDim scores
    --   target: batchSize * nDim binary tensor
    --   dim: sort indexing dimension
    --   threshold: scoring threshold
    --
    local sortPred, sortInd = torch.sort(prediction, dim, true)
    local sortTarget = target:gather(dim, sortInd):byte()
    local predBinary = torch.ge(sortPred, threshold)
    local tp = torch.cmul(predBinary, sortTarget)
    local fp = torch.eq(torch.csub(predBinary, sortTarget), 1)
    -- retrieve true positive and false positive examples
    tp = tp:cumsum(dim):float()
    fp = fp:cumsum(dim):float()
    local precision = torch.cdiv(tp, torch.add(tp, fp))
    -- retrieve true positive + false negative examples
    nInst = torch.sum(sortTarget, dim):expandAs(sortTarget):float()
    local recall = torch.cdiv(tp, nInst)

    -- bug in meanAP, accuracy instead
    local acc = precision:select(dim, tp:size(dim)):mean()
    return acc
end