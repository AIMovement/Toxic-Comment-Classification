# README #

## What is this repository for? ##

This repo is used as a collaboration platform for the Kaggle competition
[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
The programming language for the repo is Python.

## How do I get set up? ##

### Git

If developing on windows, [Git bash](https://git-scm.com/download/win) is
recommended. It includes a small Linux-like environment based on
[MSYS2](http://www.msys2.org/) (which in turn is based on
[Cygwin](https://www.cygwin.com/).)


### Python

Python requires an IDE or text editor of choice e.g.: 
* PyCharm (https://www.jetbrains.com/pycharm/) 		
* Sublime Text (https://www.sublimetext.com/) 		

Install Python 3.5.X or 3.6.X as Tensorflow requires it. 

	For Windows: 
		•	Download Windows x86-64 executable installer from [python.org](https://www.python.org/downloads/)
		•	Tick the “Add Python to PATH” option 
		•	Click “Custom Installation” and leave the first page unchanged. 
		•	Choose to install python to the directory “C:/Python3X” where X is the Python version preferred. 
		•	Click “Install” 
		•	Check if the installation was successful by opening the command prompt and type: python3 and verify that a similar response is obtained as below: 

	For OSX: 
		Follow: https://medium.com/@GalarnykMichael/install-python-on-mac-anaconda-ccd9f2014072

You also need to install `virtualenv`, as that is might not bet  installed by
default:

`pip install virtualenv`

If you do it from a VCC computer, at VCC, you will likely have a problem with
the VCC proxy. See [below](#vcc-proxy-settings) for how to work around that.

#### First time setup

Before starting development a so-called virtual environment is good to setup.
Do this by executing

`setup/setup_virtualenv.sh`

This will install some nice-to-have python modules like Keras, Tensorflow and
Matplotlib.

Now do the steps below under
[Every time initialization](#every-time-initialization)

#### Every time initialization

To activate the local python environment, go to the root of the Git repo and
execute

on Linux: `source env/bin/activate`

on Windows: `source env/Scripts/activate`

Your prompt should now show `(env)` as the first characters to the left.


## Install python packages

In order to install new python packages, run

`pip install <package>`

If you do it from a VCC computer, at VCC, you will likely have a problem with
the VCC proxy. See [below](#vcc-proxy-settings) for how to work around that.

If you want others to use the add-on, add it to the list of modules to install
by editing the file `setup/REQUIREMENTS.txt` and commit it.


## VCC proxy settings

When working from inside the VCC network a proxy is placed between the VCC
network and the rest of the internet. We need to tell the tools to use this
proxy, and not attempting to connect to the internet directly. In a terminal
run:

`export http_proxy=proxy.volvocars.net:83 https_proxy=$http_proxy`

and then retry your previous action.


## How to submit an entry
Files can be manually uploaded to the [competition
site](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge), or the
submission script can be run:

`./submit.sh <file_to_submit>`

This script will ask for a breif description of the submission and Kaggle
username and password.

		
## Contribution guidelines ##
Don't push competition data to repo. 

## Who do I talk to? ##

Slack-workspace: "Kaggle-ED"
