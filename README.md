CNN_Torch7
==========
This code use the code of Supervised Learning tutorial of Torch7. I add the loading of image by using graphicsMagick for Torch7.

1. for the code intepretation: http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_1_data, http://torch.cogbits.com/doc/tutorials_supervised/
2. How to use graphicsmagick in Torch7: https://github.com/clementfarabet/graphicsmagick
3. I modify image size to 200 * 200, then we need to change the input size of some layers in CNN.Please see the 2_model.lua
4. data/ directory:
   images/:  store the image according the label directory.
   train.txt: store the training image list.
   test.txt: store the testing image list.
   labels.txt: store the labels list.

