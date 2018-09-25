# Requirements
# - Open a PrednetClient object
#   - With this object you can download data, choose whether you want to preload a trained
#     PredNet model or build a new one, analyze the training metrics, and output the predictions
#     for given input data.
#
# Desired Behaviour
#
#   >>> data_downloader = FTPDataLoader(output="Drone")
#   >>> data_downloader.connect('host')
#   >>> data_downloader.login('usr', 'pswrd', 'acct')
#   >>> data_downloader.download(hickle_dump="drone.hkl")
#   >>> X = data_downloader.get_data(sources='2011_09_20')
#   >>> 
#   >>> client = PredNetClient()
#   >>> client.load_pretrained()
#   >>> client.predict(X[:10], verbose=1)
#
# This would then give a readout of the MSE over time of the prediction, and a plot of the predicted
# frames against the original frames.