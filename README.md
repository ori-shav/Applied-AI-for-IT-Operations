# Applied-AI-for-IT-Operations
Applied AI for IT Operations

## Use Case 1 : RCA
RCA stands for root cause analysis, and it's pretty much exactly what it sounds like. It is an iterative, interrogative technique used to explore the cause-and-effect relationships of an underlying problem. When a problem happens, what we get to see are the symptoms of the problem. The symptoms need to be mapped to a root cause by asking a series of questions and conducting a set of exploratory tests. This is similar to how a doctor diagnoses a symptom, like a fever, to a root cause, like a viral infection. How does this apply to ITOps? ITOps receives a number of service incidents every day from users. The incidents usually state the symptoms that are observed by the user. ITOps engineers may further analyze the problem to identify more symptoms. Then comes the analysis process of narrowing down the symptom to its root cause.
AI can help here by looking at the symptoms and predicting the root causes. This helps ITOps to get down to fixing the root cause quickly.

Classification is a machine learning problem of identifying a set of categories to which a new observation belongs based on a training set that contains observations for which the category is already known. Classification is the most common type of machine learning problem.

There are a number of algorithms used for classification. Those include simple decision trees, naive Bayes, random forests, support vector machines, and deep learning. There are also a number of libraries that provide implementation of these algorithms. In Python, the most popular one is scikit-learn. In recent years, deep learning has revolutionized classification with its ability to handle complex relationships and generating highly accurate models.

First we load the CSV into a Pandas data frame using the read_csv method. We then print the data types and contents of the data frame to check if the loading happened successfully. Let's execute this code. We see now the contents printed correctly. Next, we need to convert data to formats that can be consumed by Keras as Keras only consumes NumPy arrays. First, the root cause column is a text attribute. We need to convert it into a numeric value. We will use the label encoder from Scikit-learn to transform the root cause into a numeric value. Next, we need to convert the data frame into a NumPy array using the to_numpy function. We then separate the training attributes X_train into the X_train array. For the target variable, we extract it to Y_train. Then we need to use the one-hot encoding on this categorical target attribute for it to be consumed by Keras. For this, we use the utils.to_categorical function.

 First, we set up the hyperparameters for the neural network. We set epochs to 20 and a batch size of hundred. Please note that we have thousand samples in the dataset, so there will be 10 batches overall. We set verbose to be one. So we can view the details of model training. The number of output classes will be set to match the unique number of labels in the target variable, namely root cause.

  We will then try with a hidden layer size of 128. For validation we split 20% of the thousand input records to be validation data. Now we can create a sequential model. We add the first hidden layer as a dense layer with activation as RLU also called rectified linear unit. We add a second similar layer. Finally, we add a softmax layer to provide categorical labels. We then compelled the model using Adam optimizer and set the loss function as categorical, cross, and trophy. We then proceed to fit the model with X_train and Y_train in order to create the model. We print the model summary to see the model structure.

  When a new incident happen, we typically identify the symptoms of the incident first, and populate the related feature variables here, like CPU load, memory load, delays and error codes. We then pass these as an array to the model's predict classes function. This function will return a numeric value for the root cause. We then translate the numeric value into a label using the inverse transform function on the encoder. 

  ## Use Case 2: Self-Help Service Desk
  