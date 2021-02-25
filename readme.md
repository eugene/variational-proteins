## :microscope: Variational Proteins Starter Pack :school_satchel:

This is a starter pack for the **Variational Proteins** project in course [02460 - Advanced Machine Learning](https://kurser.dtu.dk/course/02460) at DTU Compute (Spring 2021). It includes the datasets and some boring boilerplate code to help with loading and parsing (`misc.py`). 

We also provide a very simple vanilla VAE (`vae.py`) and a training/eval loop (`train.py`) to get you started. If you need to brush up on your Variational Autoencoders, check out week 7 of [02456 - Deep Learning](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch).

### :train: Training 
To train the included toy VAE and see all components in action:
```
python train.py
```
On training completion a file `trained.model.pth` will be created - it will include training progress,
model parameters and other stuff ready to be explored by the `notebook.ipynb` jupyter notebook.

### :question: What's next
When you are comfortable with the data and the problem, consider working on the following ideas:

#### Ideas from the [paper](https://arxiv.org/pdf/1712.06527.pdf)
* Group Sparsity Prior (Limit the influence of neurons to a small number of positions)
* Bayesian Learning (Prevent overfitting and achieve an "ensambling" effect)
* Sequence weighting (Fix overrepresentation in the dataset)

#### Other ideas
* Different VAE architecture (eg. Hierarchical VAE)
* Compare to a GPLVM model
* Bayesian Optimization in the latent space.
