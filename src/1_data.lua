----------------------------------------------------------------------
-- This script demonstrates how to load image into Torch7
-- and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
-- Modify the code using graphicmagick: guoyilin1987@gmail.com
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require 'lfs'
gm = require 'graphicsmagick'
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Dataset Preprocessing')
  cmd:text()
  cmd:text('Options:')
  -- cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
  cmd:option('-visualize', true, 'visualize input data and weights during training')
  cmd:text()
  opt = cmd:parse(arg or {})
end

height = 200
width = 200
--see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function read_file (file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end


-- read all label name. hash them to id.
labels_id = {}
label_lines = read_file('../data/labels.txt')
for i = 1, #label_lines do
  labels_id[label_lines[i]] = i
end

-- read train data. iterate train.txt

local train_lines = read_file("../data/train.txt")
local train_features = torch.Tensor(#train_lines, 3, height, width) -- dimension: sample number, YUV, height, width
local train_labels = torch.Tensor(#train_lines) -- dimension: sample number

for i = 1, #train_lines do
  local image = gm.Image("../data/images/" .. train_lines[i])
  image:size(width, height)
  img_yuv = image:toTensor('float', 'YUV', 'DHW')
  --print(img_yuv:size())
  --print(img_yuv:size())
  train_features[i] = img_yuv
  local label_name = train_lines[i]:match("([^,]+)/([^,]+)")
  train_labels[i] = labels_id[label_name]
  --print(train_labels[i])
  if(i % 100 == 0) then
    print("train data: " .. i)
  end
end

trainData = {
  data = train_features:transpose(3,4),
  labels = train_labels,
  --size = function() return #train_lines end
  size = function() return #train_lines end
}

-- read test data. iterate test.txt
local test_lines = read_file("../data/test.txt")

local test_features = torch.Tensor(#test_lines, 3, height, width) -- dimension: sample number, YUV, height, width
local test_labels = torch.Tensor(#test_lines) -- dimension: sample number

for i = 1, #test_lines do
  -- if image size is zero, gm.Imge may throw error, we need to dispose it later.
  local image = gm.Image("../data/images/" .. test_lines[i])
  --print(test_lines[i])

  image:size(width, height)
  local img_yuv = image:toTensor('float', 'YUV', 'DHW')
  --print(img_yuv:size())
  test_features[i] = img_yuv
  local label_name = test_lines[i]:match("([^,]+)/([^,]+)")
  test_labels[i] = labels_id[label_name]
  --print(test_labels[i])
  if(i % 100 == 0) then
    print("test data: " .. i)
  end
end

testData = {
  data = test_features:transpose(3,4),
  labels = test_labels,
  --size = function() return #test_lines end
  size = function() return #test_lines end
}

trsize = #train_lines
tesize = #test_lines
----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch,
-- in general by doing: dst = src:type('torch.TypeTensor'),
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
--print '==> preprocessing data: colorspace RGB -> YUV'
--for i = 1,trainData:size() do
--   trainData.data[i] = image.rgb2yuv(trainData.data[i])
--end
--for i = 1,testData:size() do
--   testData.data[i] = image.rgb2yuv(testData.data[i])
--end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  mean[i] = trainData.data[{ {},i,{},{} }]:mean()
  std[i] = trainData.data[{ {},i,{},{} }]:std()
  trainData.data[{ {},i,{},{} }]:add(-mean[i])
  trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  testData.data[{ {},i,{},{} }]:add(-mean[i])
  testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module,
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
  for i = 1,trainData:size() do
    trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
  end
  for i = 1,testData:size() do
    testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
  end
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
  trainMean = trainData.data[{ {},i }]:mean()
  trainStd = trainData.data[{ {},i }]:std()

  testMean = testData.data[{ {},i }]:mean()
  testStd = testData.data[{ {},i }]:std()

  print('training data, '..channel..'-channel, mean: ' .. trainMean)
  print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

  print('test data, '..channel..'-channel, mean: ' .. testMean)
  print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using gfx.image().
if opt.visualize then
  first256Samples_y = trainData.data[{ {1},1 }]
  first256Samples_u = trainData.data[{ {2},2 }]
  first256Samples_v = trainData.data[{ {2},3 }]
  gfx.image(first256Samples_y, {legend='Y'})
  gfx.image(first256Samples_u, {legend='U'})
  gfx.image(first256Samples_v, {legend='V'})
end
