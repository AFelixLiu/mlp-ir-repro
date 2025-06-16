# README

This repo is a local reproduction of the paper**
*"[Machine-learning Prediction of Infrared Spectra of Interstellar Polycyclic Aromatic Hydrocarbons](https://iopscience.iop.org/article/10.3847/1538-4357/abb5b6)"
***, with code from **[here](https://zenodo.org/records/3979217)**.



## Recommended configuration

* Python == 3.10.*
* TensorFlow == 2.13.0
* CUDA == 11.8
* cuDNN == 8.6.0



## Possible problems

* Problem: `ValueError: None values not supported.`

* Solution:
    * Step 1: Comment on this line of code.
      ```text
      opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
      ```

    * Step 2: Add this line of code below.
      ```text
      opt = keras.optimizers.Adam(learning_rate=1e-4, )
      ```
