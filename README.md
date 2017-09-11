# MNIST

https://www.kaggle.com/c/digit-recognizer/data

### Best Accuracy: 99.428% ###

MNIST is a web application designed to determine what number draw a user in a drawboard.

This project has three modules:

1. #### Frontend module: ####

This module contains user interface, where a user can draw a number on a canvas element.

This module was written in:
* TypeScript
* Angular 2
* Bootstrap

2. #### Backend module: ####

This module receive the canvas image from the frontend module, save it, execute the convolutional neural network to determine what number was drew and return the response to the frontend module.

This module was written in:
* Java 1.8
* Spring Boot
* Gradle

3. #### CNN module: ####

This module is the convolutional neural network, which determine what number was drew by the user.

This module was written in:
* Pythone 3.5
* TensorFlow
* Flask
