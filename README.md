# KRFD (Kernel Regression for Functional Data)

This module (KRFD.py) provides a functional output regression model based on the kernel method called KRFD. A functional output regression model is a regression model in which the output is expressed in functional form, as opposed to the general regression model in which scalar or vector values are output. 

Unlike functional output regression models, which require the functionalization of data, KRFD constructs a model directly from vector input values and discrete functional output values. If the functional output values are observed at the same measurement point set for each input, use the KRFD_model class in the KRFD.py module. If the functional output values are observed at different measurement point sets for each input, use the KRSFD_model class. Furthermore, since this model is Bayesianized, users can perform analytical quantification of prediction uncertainty and sampling of the predicted functions for any given input, in both the KRFD_model and KRSFD_model classes. 

Using tutorial_KRFD.ipynb and tutorial_KRSFD.ipynb files, users can perform functional output regression using KRFD on artificial data. A paper on KRFD is in preparation and will be uploaded to arXiV by October 2024.

# Usage
 
1. First install the dependencies listed below.

2. Clone the `KRFD` github repository:
```bash
git clone https://github.com/Minoru938/KRFD.git
```

3. `cd` into `KRFD` directory.

4. Run `jupyter notebook` and open `tutorial_KRFD.ipynb` to demonstrate in `KRFD_model` class in the KRFD.py module.

5. For the `KRSFD_model` class tutorial, please run `tutorial_KRSFD.ipynb`. 


# Dependencies

For KRFD.py

* pandas version = 1.5.1
* numpy version = 1.22.4
* scipy version == 1.8.1

For tutorial_KRFD.ipynb and tutorial_KRSFD.ipynb

* optuna version == 3.0.3
* joblib version == 1.2.0 
* matplotlib version == 3.7.1  
* scikit-learn version == 1.1.3  
