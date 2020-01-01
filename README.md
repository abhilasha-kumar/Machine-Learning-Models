# Machine-Learning-Models
A collection of machine learning algorithms I wrote. These files include algorithms to implement the following models:

### Perceptron Learning Algorithm 

The Perceptron Learning Algorithm is a simple classifier that separates any two classes using a simple iterative approach to updating the weight vector. Starting with a weight vector initialized to 0s, it randomly picks a misclassified example and updates it weights and recomputes the classification. This process is repeated until there is no misclassified example -- assuming that the data is linearly separable.

### Logistic Regression Model

The logistic regression model uses full gradient descent and different learning rates and maximum iterations to update the weights on training and testing data. The reg_z file examines the logistic regression model for normalized predictors.

### Bagged Decision Trees

The bagged decision tree model uses a part of the MNIST dataset to classify the digits 1, 3, and 5. The out-of-bag error is calculated as a function of total bags on training and testing data. 

### Random Forests

The random forest decision tree model uses a part of the MNIST dataset to classify the digits 1, 3, and 5. The out-of-bag error is calculated as a function of total bags on training and testing data. 

### Adaboost Algorithm

The Adaboost algorithm is implemented to reduce the bias in the decision tree model. 

### Support Vector Machine Algorithm 

The SVM model is implemented on a simple linear classification 2-D problem.
