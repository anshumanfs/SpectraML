# Glossary of Machine Learning Terms

[← Back to Index](index.md)

This glossary provides definitions for common terms used in machine learning and spectral data analysis.

## A

**Accuracy**  
The proportion of correct predictions made by a model out of all predictions made.

**Activation Function**  
A mathematical function applied to transform the output of a neural network node. Common functions include ReLU, sigmoid, and tanh.

**Algorithm**  
A step-by-step procedure for solving a problem, often referring to the specific mathematical approach used by a machine learning model.

**Anomaly Detection**  
The identification of rare items, events, or observations that differ significantly from the majority of the data.

**Area Under the Curve (AUC)**  
A performance measurement for classification problems that represents the area under the ROC curve.

**Attention Mechanism**  
A technique in neural networks that allows the model to focus on specific parts of the input when producing an output.

**Autoencoder**  
A type of neural network designed to learn efficient representations of data by training the network to ignore signal noise.

## B

**Backpropagation**  
An algorithm for calculating gradients in neural networks, working backwards from the output layer to update weights.

**Bagging (Bootstrap Aggregating)**  
A technique that combines multiple models trained on different bootstrapped samples of the dataset.

**Baseline Correction**  
A preprocessing step for spectral data that removes background interference.

**Batch Size**  
The number of training examples used in one forward/backward pass of gradient descent.

**Bias**  
1. The tendency of a model to consistently learn the same wrong patterns.
2. A parameter in neural networks added to the weighted input of a neuron.

**Boosting**  
An ensemble technique where models are trained sequentially, with each new model focusing on the errors of previous models.

## C

**Classification**  
A supervised learning task where the model predicts categorical labels.

**Clustering**  
An unsupervised learning task that groups similar instances together.

**Confusion Matrix**  
A table used to describe the performance of a classification model by showing the counts of true positives, false positives, true negatives, and false negatives.

**Convolutional Neural Network (CNN)**  
A type of neural network with specialized layers designed to process grid-like data such as images.

**Cross-Validation**  
A technique for evaluating how well a model will generalize to new data by partitioning the original dataset.

**Curse of Dimensionality**  
The problems that arise when analyzing data in high-dimensional spaces that do not occur in low-dimensional spaces.

## D

**Data Augmentation**  
Techniques to increase the diversity of training data without collecting new data.

**Data Drift**  
Changes in the distribution of data over time that may necessitate model retraining.

**Decision Boundary**  
The hypersurface that separates different class regions in the feature space.

**Deep Learning**  
A subset of machine learning using neural networks with many layers to learn representations of data.

**Dimensionality Reduction**  
The process of reducing the number of random variables under consideration.

**Dropout**  
A regularization technique where randomly selected neurons are ignored during training.

## E

**Early Stopping**  
A form of regularization used to avoid overfitting by stopping training when performance on a validation dataset starts to degrade.

**Embedding**  
A mapping from discrete objects (like words) to vectors of continuous numbers.

**Ensemble Method**  
A technique that combines multiple models to improve performance.

**Epoch**  
One complete pass through the entire training dataset.

**Evaluation Metric**  
A measure used to assess the performance of a machine learning model.

## F

**F1 Score**  
The harmonic mean of precision and recall, providing a balance between the two.

**Feature**  
An individual measurable property or characteristic of a phenomenon being observed.

**Feature Engineering**  
The process of using domain knowledge to create features that make machine learning algorithms work better.

**Feature Importance**  
A score assigned to input features based on how useful they are at predicting a target variable.

**Fine-Tuning**  
The process of taking a pre-trained model and adapting it to a new, similar task.

## G

**Gradient Descent**  
An optimization algorithm for finding the minimum of a function.

**Gradient Boosting**  
An ensemble technique that builds models sequentially, each trying to correct errors from the previous one.

**Ground Truth**  
The accurate, verified data used as a reference for training and testing models.

## H

**Hyperparameter**  
A parameter whose value is set before the learning process begins, as opposed to parameters learned during training.

**Hyperparameter Tuning**  
The process of finding the optimal hyperparameters for a learning algorithm.

## I

**Imbalanced Data**  
A dataset where the classes are not represented equally.

**Inference**  
The process of using a trained model to make predictions on new data.

**Information Gain**  
A measure of the reduction in entropy achieved by splitting the data according to a specific feature.

## K

**K-Fold Cross-Validation**  
A cross-validation technique where the dataset is divided into k subsets, with each subset serving as the test set once.

**K-Means Clustering**  
An unsupervised learning algorithm that partitions observations into k clusters.

**K-Nearest Neighbors (KNN)**  
A simple, instance-based learning algorithm that classifies new data points based on their similarity to known examples.

**Kernel**  
A function used in SVM algorithms to map data from a low-dimensional space to a higher-dimensional space.

## L

**Label**  
The target variable or outcome in supervised learning.

**Learning Rate**  
A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

**Loss Function**  
A function that measures the difference between the model's predictions and the actual values.

**LSTM (Long Short-Term Memory)**  
A type of recurrent neural network architecture designed to handle long-term dependencies.

## M

**Machine Learning**  
A field of study that gives computers the ability to learn without being explicitly programmed.

**Matrix Factorization**  
A technique to decompose a matrix into the product of multiple matrices, often used in recommendation systems.

**Mean Absolute Error (MAE)**  
The average of absolute differences between predictions and actual observations.

**Mean Squared Error (MSE)**  
The average of the squared differences between predictions and actual observations.

**Model**  
A mathematical representation of a real-world process, learned from data.

**Multicollinearity**  
A phenomenon in which two or more predictor variables in a model are highly correlated.

## N

**Neural Network**  
A computing system inspired by biological neural networks that forms the basis of many deep learning algorithms.

**Normalization**  
The process of scaling individual samples to have unit norm.

**Null Accuracy**  
The accuracy that could be achieved by always predicting the most frequent class.

## O

**One-Hot Encoding**  
A process that converts categorical variables into a form that could be provided to machine learning algorithms.

**Optimization**  
The process of finding the best solution to a problem based on some criteria.

**Outlier**  
A data point that differs significantly from other observations.

**Overfitting**  
A modeling error that occurs when a model learns the training data too well, including its noise and outliers.

## P

**Partial Least Squares (PLS)**  
A method that relates two data matrices by finding the multidimensional direction that explains the maximum covariance between them.

**Precision**  
The ratio of true positives to the sum of true positives and false positives.

**Principal Component Analysis (PCA)**  
A technique for dimensionality reduction that finds the directions of maximum variance.

**Pruning**  
The process of reducing a neural network's size by removing connections or neurons based on some criteria.

## R

**Random Forest**  
An ensemble learning method that constructs multiple decision trees during training.

**Recall**  
The ratio of true positives to the sum of true positives and false negatives.

**Receiver Operating Characteristic (ROC) Curve**  
A graphical plot that illustrates the diagnostic ability of a binary classifier system.

**Recurrent Neural Network (RNN)**  
A type of neural network designed for sequential data by allowing outputs from previous time steps to affect current time steps.

**Regression**  
A supervised learning task where the model predicts continuous values.

**Regularization**  
Techniques used to prevent overfitting by adding a penalty term to the loss function.

**Reinforcement Learning**  
A type of machine learning where an agent learns to behave in an environment by performing actions and receiving rewards.

## S

**Semi-Supervised Learning**  
A learning paradigm concerned with the study of how to combine labeled and unlabeled data to improve learning.

**Spectral Data**  
Data that represents the intensity of a signal as a function of wavelength, frequency, or energy.

**Standardization**  
The process of rescaling features to have zero mean and unit variance.

**Stochastic Gradient Descent (SGD)**  
A variant of gradient descent that uses only a subset of the data at each iteration.

**Supervised Learning**  
A task where the model learns from labeled training data.

**Support Vector Machine (SVM)**  
A supervised learning algorithm that can be used for classification or regression tasks.

## T

**Test Set**  
A subset of the data used to test the model after training.

**Training Set**  
A subset of the data used to train the model.

**Transfer Learning**  
A technique where a model developed for one task is reused as the starting point for a model on a second task.

**True Positive Rate**  
The ratio of true positives to the sum of true positives and false negatives (also known as recall).

## U

**Underfitting**  
A modeling error that occurs when a model is too simple to capture the underlying pattern of the data.

**Unsupervised Learning**  
A task where the model learns patterns from unlabeled data.

**Upsampling**  
A technique to address class imbalance by increasing the number of instances in the minority class.

## V

**Validation Set**  
A subset of the data used to tune hyperparameters and provide an unbiased evaluation during training.

**Vanishing Gradient Problem**  
A difficulty found in training neural networks with gradient-based methods and backpropagation.

**Variance**  
The sensitivity of a model to fluctuations in the training data.

## W

**Weights**  
Parameters within a neural network that transform input data within the network's layers.

**Word Embedding**  
A technique in natural language processing where words are represented as dense vectors in a continuous vector space.

## X

**XGBoost**  
An efficient and scalable implementation of gradient boosting.

## Y

**Yield**  
In spectroscopy, the conversion of absorbed energy to emission.

## Z

**Zero-Shot Learning**  
A problem setup where a learner observes samples from certain classes during training but must recognize objects from unseen classes at test time.

---

## Navigation

**Next**: [Code Examples](code_examples.md)  
**Previous**: [Best Practices](best_practices.md)  

**Related Topics**:
- [Introduction to Machine Learning](introduction.md)
- [Model Selection](model_selection.md)
- [Feature Engineering](feature_engineering.md)
