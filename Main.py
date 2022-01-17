'''Operation: Pancake swap guess
Goal: 5 Minute predictions for pancakeswap prediction game
Needed: Train the network on ETH-USDT dataset
Required: New model training module, with data batch-load to avoid loading whole set into RAM
Needed: Transfer learning to BNB-USDT dataset
Required: Coingecko API hookup (Going to use a different one)
          1 minute to 5 minute candle conversions (Can we make it universal, from 1 minute to any interval?)
Required: Test hand-made dataset to test neural network
Required: Shift the 5m dataset so it alligns with pancake swap one'''




'''From Paper what good practices should be: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
Shuffle the examples - Feed various companies/markets to the training
Centre the input variables by substracting the mean - SS transform
Normalize input variables to a standard deviation of 1 - SS transform?
If possible, decorrelate input variables - PCA (Look into whiten = true in sklearn library)?
Pick a network with sigmoid function (Fig 4 in paper) - Use sigmoid activation
Set the target values within range of the sigmoid, typically -1 to 1 - Minmax scaler
Initialize weights to random values - (Maybe add a genetic algo too for mutating weights?)
'''
