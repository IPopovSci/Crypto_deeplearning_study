
#Todo: implement denoisining, AlphaLense to determine which TA factors are more useful
'''From Paper what good practices should be: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
Shuffle the examples - Feed various companies/markets to the training
Centre the input variables by substracting the mean - SS transform
Normalize input variables to a standard deviation of 1 - SS transform?
If possible, decorrelate input variables - PCA (Look into whiten = true in sklearn library)?
Pick a network with sigmoid function (Fig 4 in paper) - Use sigmoid activation
Set the target values within range of the sigmoid, typically -1 to 1 - Minmax scaler
Initialize weights to random values - (Maybe add a genetic algo too for mutating weights?)
'''

#Todo: To bring the project to presentable form -
#Documentation
#Tests
#