### Implementation notes

1) Since we want all colors' embeddings attending each other, we use all inputs in one batch. This is different from the
   usual way of training a model, where we would have multiple batches of data.
2) The first testing of the model I've done using 2 classes, 1000 vertices and 100 colors.

### Twitch run
There is no reason that the attention model will 
succeed classifying the test color since its embedding is
random. 

train02 and train03 are training and HP tuning runs for the twitch dataset.
The best params for twitch are saved in results twitch.

train04.py is the training for the new data (VAE-KNN)

Yoram's innovations:
* train05.py is the training code for the weight in the BCE.
* train06.py is the training code for subsampling. 
* train07.py is the training code for the new edges and RGCN.