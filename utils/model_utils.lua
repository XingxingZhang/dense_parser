
local model_utils = {}

function model_utils.combine_selectnet_parameters(forward_lstm, backward_lstm, attention)
    --This is a method only works for fflstm--
    -- get parameters. Note we will ignore the lookup table "ngram_lookup"
    local parameters = {}
    local gradParameters = {}
    
    -- get LSTM parameters. IGNORE ngram_lookup LookupTable
    local function getLSTMParameters(fflstm, parameters, gradParameters)
      
      for _, node in ipairs(fflstm.forwardnodes) do
        -- check IF this is a module and the module is not a lookup table
        if node.data.module then
          -- 'enc_ngram_lookup' or node.data.annotations.name == 'dec_ngram_lookup'
          if node.data.annotations.name ~= 'backward_lookup' then
            local mp,mgp = node.data.module:parameters()
            if mp and mgp then
              for i = 1,#mp do
                table.insert(parameters, mp[i])
                table.insert(gradParameters, mgp[i])
              end
            end
          else
            print('[combine_selectnet_parameters] found backward_lookup! ' .. node.data.annotations.name)
          end
        end
      end
      
    end
    
    getLSTMParameters(forward_lstm, parameters, gradParameters)
    getLSTMParameters(backward_lstm, parameters, gradParameters)
    getLSTMParameters(attention, parameters, gradParameters)
    
    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end


function model_utils.combine_selectnet_pos_parameters(forward_lstm, backward_lstm, attention)
    --This is a method only works for fflstm--
    -- get parameters. Note we will ignore the lookup table "ngram_lookup"
    local parameters = {}
    local gradParameters = {}
    
    -- get LSTM parameters. IGNORE ngram_lookup LookupTable
    local function getLSTMParameters(fflstm, parameters, gradParameters)
      
      for _, node in ipairs(fflstm.forwardnodes) do
        -- check IF this is a module and the module is not a lookup table
        if node.data.module then
          -- 'enc_ngram_lookup' or node.data.annotations.name == 'dec_ngram_lookup'
          if node.data.annotations.name ~= 'backward_lookup' and node.data.annotations.name ~= 'backward_pos_lookup' then
            local mp,mgp = node.data.module:parameters()
            if mp and mgp then
              for i = 1,#mp do
                table.insert(parameters, mp[i])
                table.insert(gradParameters, mgp[i])
              end
            end
          else
            print('[combine_selectnet_parameters] found backward_lookup! ' .. node.data.annotations.name)
          end
        end
      end
      
    end
    
    getLSTMParameters(forward_lstm, parameters, gradParameters)
    getLSTMParameters(backward_lstm, parameters, gradParameters)
    getLSTMParameters(attention, parameters, gradParameters)
    
    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

-- share parameters of two lookupTables: forward lstm and backward lstm
function model_utils.share_fflstm_lookup(fflstm)
  local lstm_lookup, ngram_lookup
  for _, node in ipairs(fflstm.forwardnodes) do
    if node.data.module then
      if node.data.annotations.name ~= nil and node.data.annotations.name:ends('ngram_lookup') then
        ngram_lookup = node.data.module
        print('[model_utils.share_fflstm_lookup] ngram_lookup found!')
      elseif node.data.annotations.name ~= nil and node.data.annotations.name:ends('lstm_lookup') then
        lstm_lookup = node.data.module
        print('[model_utils.share_fflstm_lookup] lstm_lookup found!')
      end
    end
  end
  
  ngram_lookup.weight:set(lstm_lookup.weight)
  ngram_lookup.gradWeight:set(lstm_lookup.gradWeight)
  
  collectgarbage()
  
  return lstm_lookup, ngram_lookup
end

function model_utils.load_embedding_init(emb, vocab, embedPath)
  require 'wordembedding'
  local wordEmbed = WordEmbedding(embedPath)
  wordEmbed:initMat(emb.weight, vocab)
  wordEmbed:releaseMemory()
  vocab = nil
  wordEmbed = nil
  collectgarbage()
end

function model_utils.load_embedding_fine_tune(emb, vocab, embedPath, ftFactor)
  require 'wordembedding_ft'
  local wordEmbed = WordEmbeddingFT(embedPath)
  local mask = wordEmbed:initMatFT(emb.weight, vocab, ftFactor)
  emb:setUpdateMask(mask)
  wordEmbed:releaseMemory()
  vocab = nil
  wordEmbed = nil
  collectgarbage()
end


function model_utils.clone_many_times_emb_ft(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    
    local master_map = BModel.get_module_map(net)
    local lt_names = {}
    for k, v in pairs(master_map) do
      if k:find('lookup') ~= nil then
        table.insert(lt_names, k)
      end
    end
    
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end
        
        local clone_map = BModel.get_module_map(clone)
        for _, k in ipairs(lt_names) do
          if clone_map[k].updateMask then
            clone_map[k].updateMask:set( master_map[k].updateMask )
          end
        end
        
        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.copy_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

return model_utils
