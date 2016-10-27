require 'nn'
local Set = require 'pl.Set';
local RandProjCrossEntropyCriterion, Criterion = torch.class('nn.RandProjCrossEntropyCriterion', 'nn.Criterion')

function RandProjCrossEntropyCriterion:setk(k)
	self.k = k
end

function RandProjCrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.ClassNLLCriterion(weights)
   self.rand_proj_num = self.k --random projection number
   self.target_table = {} --save the different classes in the minibatch
   self.target_proj = torch.Tensor() --save the new target, the length should be equal to the batchsize
   self.rand_proj_table = {} --save the rand_proj classes
end

function RandProjCrossEntropyCriterion:updateOutput(input, target)
	input = input:squeeze()
	target = type(target) == 'number' and target or target:squeeze()
	self.rand_proj_num = self.k --random projection number
--clear the target_table in the begining of each training iteration
	self.target_table = {}
--pick out different classes in a mini-batch
	local target_set = Set(torch.totable(target))
	for k,v in pairs(target_set) do
	   table.insert(self.target_table, k)
	end
--old_new_target_map is used for old new classese mapping
	local old_new_target_map = {}
	for k,v in ipairs(self.target_table) do
		old_new_target_map[v] = k
	end
--set new target label of all of the minibatch samples
	local pos_cls_table = {}
	for i = 1, input:size(1) do
		table.insert(pos_cls_table, old_new_target_map[target[i]])
	end
--creat new label
	self.target_proj = torch.Tensor(pos_cls_table)
--the random selection should involve postive classes in a mini-batch, so we randomly pickout the negative classes in the rest of classes
	local rand_select_num = self.rand_proj_num - input:size(1)
	local neg_cls_table = {}
	local hole_cls = torch.range(1,input:size(2)):totable()
	local res_set = Set.difference(Set(hole_cls), target_set)
	local res_table = {}
	for k,v in pairs(res_set) do
		table.insert(res_table, k)
	end
--negative classes are picked out in the rest of classes, and they are not repeated
	res_table_random_index = torch.randperm(#res_table)
	for i = 1,rand_select_num do
		table.insert(neg_cls_table, res_table[res_table_random_index[i]])
	end
--creat rand_proj_table, put the sample which are coming from positive classes in the front, then follow the negative classes
	self.rand_proj_table = {}
	for i = 1, #pos_cls_table do
		table.insert(self.rand_proj_table, pos_cls_table[i])
	end
	for i = 1, #neg_cls_table do
		table.insert(self.rand_proj_table, neg_cls_table[i])
	end
--project the input to the random select space
	local rand_filter_input = torch.zeros(input:size(1), self.rand_proj_num):cuda()
	for k,v in ipairs(self.rand_proj_table) do
		rand_filter_input[{{},k}] = input[{{},v}]
	end
--compute the loss
	self.lsm:updateOutput(rand_filter_input)
	self.nll:updateOutput(self.lsm.output, self.target_proj:cuda())
	self.output = self.nll.output
	return self.output
end

function RandProjCrossEntropyCriterion:updateGradInput(input, target)
	local size = input:size()
	input = input:squeeze()
	target = type(target) == 'number' and target or target:squeeze()
--give the rand_sample_input
	local rand_filter_input = torch.zeros(input:size(1), self.rand_proj_num):cuda()
	for k,v in ipairs(self.rand_proj_table) do
		rand_filter_input[{{},k}] = input[{{},v}]
	end
--compute the gradient which gernerated by sampled classes
	self.nll:updateGradInput(self.lsm.output, self.target_proj:cuda())
	self.lsm:updateGradInput(rand_filter_input, self.nll.gradInput)
--generate a matrix to save the gradient
	local rand_proj_grad = torch.zeros(input:size(1), input:size(2)):cuda()
	for k,v in ipairs(self.rand_proj_table) do
		rand_proj_grad[{{},v}] = self.lsm.gradInput[{{},k}]
	end
--turn back the gradient
	self.gradInput:view(rand_proj_grad, size)
	return self.gradInput
end

return nn.CrossEntropyCriterion
