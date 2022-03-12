'''Operation: Pancake swap guess
Goal: 5 Minute predictions for pancakeswap prediction game
Needed: Train the network on ETH-USDT dataset
Required: New model training module, with data batch-load to avoid loading whole set into RAM
Needed: Transfer learning to BNB-USDT dataset
Required: Coingecko API hookup (Going to use a different one)
          1 minute to 5 minute candle conversions (Can we make it universal, from 1 minute to any interval?)
Required: Test hand-made dataset to test neural network
Required: Shift the 5m dataset so it alligns with pancake swap one'''

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
# 1) Get data - introduce a home path via .env, separate data into ticker/ticker_{timeinterval}.csv format
# 2) Create training mode, which will grab only CSV data, and prediction mode - which will use one of the APIS to grab only the latest data (Make sure that it grabs the data of the right interval)
# 3) Modify the rescaler function - Take an interval and save a new csv under home/ticker/ticker_{new_timeinterval}.csv format - not for pipeline, utility
# 4) No need for pancake mode, since training is only done on CSV and inference/predictions only done API data
# 5) Write 5 models - Dense, LSTM w/ Attention, Conv1d, Conv1dLSTM and Conv1dLSTM w/ conv2d as final layer. Train models.
# 6) Create information coefficient computation w/ graphs, as well as returns using alpha library as per book
# 7) Separate training/prediction functions, that load in correct callbacks functions for each network (Based on Alons work)