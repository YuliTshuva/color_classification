### Implementation notes

1) Since we want all colors' embeddings attending each other, we use all inputs in one batch. This is different from the
   usual way of training a model, where we would have multiple batches of data.
2) The first testing of the model I've done using 2 classes, 1000 vertices and 100 colors.