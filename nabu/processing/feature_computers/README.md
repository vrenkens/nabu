# Feature Computers

a feature computer is used to compute audio features. To create your own feature
computer you can inherit from the general FeatureComputer class defined in
feature_computer.py and overwrite the abstract methods.
Afterwards you should add it to the factory method in
feature_computer_factory.py and to the package in \_\_init\_\_.py.
