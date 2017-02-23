### Deep Neural Network Visualization

This is a toy example for visualization of deep neural network's layer
activation. The DNN trained on MNIST training data.

1) MNIST database downloaded and fetched from the web

2) A convolutional neural network trained on the entire database consist of
   three convolutional layer with kernel size: 5 followed by a max pool layer
   with kernel size: 2. The first and second convolutional layer performed 5
   times on it's input, the third one performed 20 times, followed by a fully
   connected layer in the end. The loss function is cross entropy based, and
   finally the optimizer performed by Adam optimizer.

3) A visualization test has been performed by selecting five test images
  from MNIST randomly and plotting the corresponding activation of every
  hidden layer.

Note that the visualization module first show the test image, then the
activations of the layers. 5 second delay has been used between image
plotting.

#![Demonstration of an activation Visualization](./doc/vis.png)
#![Demonstration of an activation Visualization](./doc/vis.png)


### Dependencies
This code is dependent on TensorFlow and Matplotlib.
It has been tested on MacOs Sierra 10.12.3 with
- Python (2.7)
- Matplotlib (1.3.1)

### Install
- *MacOs:*

    Install TensorFlow, Matplotlib

    ## Open a terminal.

    Open `Terminal`. This tutorial assumes you are using `bash`, which you
    probably are.

    ## Clone this repository

    Using git, clone this tutorial and enter that directory.

    ```
    git clone https://github.com/attilaborcs/dnn-visualization.git
    cd dnn-visualization
    ```

    ## Install Pip and Virtualenv

    Pip is a package management system used to install and manage software
    packages written in Python.  Virtualenv allows you to manage multiple
    package installations.

    At your Terminal window, run the following command.
    ```
    # Mac OS X
    sudo easy_install --upgrade pip
    ```

    Once you've installed pip, you'll need to add a few more packages.

    ```
    sudo easy_install --upgrade six
    sudo pip install --upgrade virtualenv
    ```

    These should some dependencies and Virtualenv.

    Now, create a virtual environment.

    ```
    virtualenv --system-site-packages ~/tensorflow
    ```

    > Note: If you have already installed anaconda, some versions of
    > anaconda and virtualenv are not compatible.  If you have trouble,
    > such as seeing errors about "sys.prefix", you may want to try to
    > use the [TensorFlow anaconda installation instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#anaconda-installation).
    > You'll also need to install matplotlib and Pillow as well to get the full experience.

    You will need to Activate the environment, which is to say switch your
    Python enviroment to a fresh one with clean dependencies.

    ```
    source ~/tensorflow/bin/activate
    ```

    You are now running in a special Python enviroment with safe
    dependencies. Your prompt should start with `(tensorflow) $`.

### Usage
You can try DNN visualization of mnist dataset located in the ./mnist/ folder by running:
```
cd dnn-visualization; python run.py
```

### Contributors
Attila Borcs
