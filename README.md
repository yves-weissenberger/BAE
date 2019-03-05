# Reference implementation of the Behavioural Autoencoder
----

## Installation

This package implements the augmented latent-variable model  (see bioarxiv link). It runs in Python 3 and assumes installation of python 3.6 as well as dependencies specified in requirements.txt. The easiest way to do this is install the anaconda python distribution and then the correct tensorflow version.


To install, download the package and, in terminal, 'cd' to the the relevant folder. We recommend running in a virtual environment. The easiest way to do this is install anaconda

Then type

     conda create -n <envName> python=3.6

where you should replace <envName> with whatever you want the environment to be called. Then activate the virtual environment by running either (depending on your system the requirements will differ):

     conda activate <envName>

or
     activate <envName>

Then install the dependencies by running:

     conda install --yes --file requirements.txt

If you plan to use the tutorial, also install Jupyter

     conda install jupyter


Then, install the package by running: 


     pip install .

###


## Tutorial

To access the tutorial, in the directory of the package, run:

    ipython notebook


Then nativate to the notebooks folder and run through the tutorial
