# ConVAE_Jet

How to improve the performance of the VAE for jet generation:

- Use diferentiable EMD as the jets reconstruction loss term in the loss function (the EMD is being used as an evaluation metric only);
- Use some rigorous optimization method to find the best hyperparameters: batch size,number of layers, layers size (conv, fc and latent), dropout and/or pool, learning rate, error function weights;
- Use normalizing flows
