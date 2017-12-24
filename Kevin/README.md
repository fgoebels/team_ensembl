Single RNN network approach gets 0.047 on public LB.  Predictions on entire training set as out-of-fold chunks from cross-validation is saved so that anyone can easily stack this method with another without needing to run code.

Next on TODO list: 

1) Clean training/test comments (autocorrection) and re-train network.
2) Develop badwords count vectors as comment encoding and try to train a RF/XGB model on this.
