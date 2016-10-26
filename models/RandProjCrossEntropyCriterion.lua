require 'nn'
local Set = require 'pl.Set';
local RandProjCrossEntropyCriterion, Criterion = torch.class('nn.RandProjCrossEntropyCriterion', 'nn.Criterion')

function RandProjCrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.ClassNLLCriterion(weights)
   self.rand_proj_num = 15000 --random projection number
   self.target_table = {}
   self.target_proj= torch.range
end

function RandProjCrossEntropyCriterion:updateOutput(input, target)
   input = input:squeeze()
   target = type(target) == 'number' and target or target:squeeze()
--clear the target_table in the begining of each training iteration
   self.target_table = {}
--pick out different classes in a mini-batch
   local target_set = Set(torch.totable(target))
   for k,v in pairs(target_set) do
       table.insert(self.target_table, k)
   end
   local pos_target_num = #self.target_table
--the random selection should involve postive classes in a mini-batch, so we randomly pickout the negative classes in the rest of classes
   local rand_select_num = self.rand_proj_num - #target_set
   local neg_cls_table = {}
   local hole_cls = torch.range(1,input:size(2)):totable()
   local res_set = Set.difference(Set(hole_cls), target_set)
   local res_table = {}
   for k,v in pairs(res_set) do
       table.insert(res_table, k)
   end
--when pickout the negative classes, any of them should not be repeated
   for i = 1,rand_select_num do
       local index = math.random(#res_table)
       table.insert(neg_cls_table, res_table[index])
       table.remove(res_table, index)
   end
--concat the target_table and negative table, the positive classes are arranged from 1 to #postive_num automatically
   for i = 1, #neg_cls_table do
       self.target_table[#self.target_table + 1] = neg_cls_table[i]
   end
--generate a new zero matrix to save the random projection input
   local rand_filter_input = torch.zeros(input:size(1), self.rand_proj_num):cuda()
   for k,v in ipairs(self.target_table) do
       rand_filter_input[{{},k}] = input[{{},v}]
   end
--generate new labels
   self.target_proj = torch.range(1,pos_target_num)
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
   for k,v in ipairs(self.target_table) do
       rand_filter_input[{{},k}] = input[{{},v}]
   end
--compute the gradient which gernerated by sampled classes
   self.nll:updateGradInput(self.lsm.output, self.target_proj)
   self.lsm:updateGradInput(rand_filter_input, self.nll.gradInput)
--generate a matrix to save the gradient
   local rand_proj_grad = torch.zeros(input:size(1), input:size(2)):cuda()
   for k,v in ipairs(self.target_table) do
       rand_proj_grad[{{},v}] = self.lsm.gradInput[{{},k}]
   end
--turn back the gradient
   self.gradInput:view(rand_proj_grad, size)

   return self.gradInput
end

return nn.CrossEntropyCriterion
