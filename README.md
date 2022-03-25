# ConVAE_Jet

How to improve the performance of the VAE for jet generation:

- Use diferentiable EMD as the jets reconstruction loss term in the loss function (the EMD is being used as an evaluation metric only);
- Use some rigorous optimization method to find the best hyperparameters: batch size, number of layers, layers size (conv, fc and latent), dropout and/or pool, learning rate, error function weights;
- Use normalizing flows

The last commit runs with PyTorch version until 1.8.0. With version 1.10.2, it is necessary to use tag v2.0.

Action items:
- Solve heritage of the classes
- Shorten *try except* block in ´optim.py´
- Create optimization flag in ´optim.py´ and ´config´
- Check the passing of the model to the compute_loss function in ´vae.py´
