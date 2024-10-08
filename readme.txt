
About this file:

This file contains a module (KRFD.py) that provides a functional output regression model based on the kernel method called KRFD. KRFD.py is mainly made up of the KRFD_model and KRSFD_model classes. Use the KRFD_model class, if the functional output values are observed at the same measurement point set for each input. Use the KRSFD_model class, if the functional output values are observed at different measurement point sets for each input.

Usage:

Set the KRFD file as the current directory, then open tutorial_KRFD.ipynb with jupyter notebook to start a tutorial that explains how to use the KRFD_model class in the KRFD.py module. For the KRSFD_model class tutorial, please run tutorial_KRSFD.ipynb. 


Dependencies:

・ For KRFD.py 

pandas version = 1.5.1
numpy version = 1.22.4
scipy version == 1.8.1

・ For tutorial_KRFD.ipynb and tutorial_KRSFD.ipynb

optuna version == 3.0.3
joblib version == 1.2.0 
matplotlib version == 3.7.1  
scikit-learn version == 1.1.3  


Environment of author:

Python 3.9.16
macOS Sonoma 14.6.1
