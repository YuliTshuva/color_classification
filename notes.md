The attention doesn't work so well. I need to drop it from train07.py. 
Before I do so, I should explain why it didn't work backing every statement with the exported files. 


Direction I should try:
1) Aggregate the color embedding of each node its embedding from the data.
2) Try the RGCN approach as well where the second kind of edges is inner color relations - edges of second kind between nodes of the same color (and then there won't be embedding for the colors - just the features in the data).
