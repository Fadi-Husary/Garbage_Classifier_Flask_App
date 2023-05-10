# Garbage_Classifier_Flask_App

This project is a garbage classifier web app. I used a Convolutional Neueral Network(CNN) model with tensorflow and keras. I made the web app using flask and deployed the CNN model in it. When the flask app is working, you will be able to upload an image of garbage and get back a clasification of the image, and the level of confidence in the accuracy of the output.

The 12 categories of garbage are: 
- paper
- cardboard
- plastic
- metal
- battery
- shoes
- clothes
- green-glass
- brown-glass
- white-glass
- biological

## Data 
The images used to train and test the model are from [kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification).

## Steps 
- Download images and packages(listed in the next section).
- Create a CNN model and save it:
-       Preproccess: Clean your data and transform it into a form that the model can use. This involves resizing images to a consistent size, normalizing pixel values, and splitting  data into training and testing sets.
-       Model Architecture: This involves choosing the number of layers in your network, the types of layers, and the parameters for each layer (like the filter    size and stride for convolutional layers).
-      Compile the Model: Specify additional training parameters such as the optimizer, loss function, and metrics for evaluating model performance.
-      Train, test, and evaluate the model.
-      Model Tuning(if needed)
-      Save the model: use 'model.save('model_path.h5') to save the whole model.
- Create a flask web app.
- Load your model into flask.
- Test the final product.


## Packages
- python
- pandas
- tensorflow
- keras
- numpy
- seaborn
- matplotlib.pyplot
- zipfile
- sys
- os
- re
- io
- flask:
-       request
-       Flask
-       render_template
- keras.layers:
-       Input
-       Conv2D
-       Dense
-       Flatten
-       MaxPooling2D
-       Input
-       GlobalAveragePooling2D
-       Dropout
-       Activation
-       BatchNormalization
-       Lambda
- keras.models:
-       Model
-       Sequential
- keras.utils:
-       to_categorical
- tensorflow.keras.models:
-       load_model
- sklearn.model_selection:
-       train_test_split
- sklearn.metrics:
-       classification_report
- PIL:
-        Image
- tensorflow.keras.preprocessing.image:
-       ImageDataGenerator
- tensorflow.keras.preprocessing.image:
-       load_img
-       img_to_array
- keras.applications:
-       xception
