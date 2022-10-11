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
   We will use a data set that contains a list of FAQ articles and a corresponding set of questions. The FAQ articles can have multiple questions associated with them. We will then build a natural language model that can take a new user question and find the closest question in the dataset.

   Latent semantic analysis (LSA) and latent semantic indexing (LSI)

   Machine learning algorithms work only with numeric data. They do not understand text. One of the most recent and popular techniques to convert text into its numeric representation is called Latent Semantic Analysis or LSA. It can use the vectorized representation of documents to analyze relationships and arrive at a similarity model. It builds an index using the latent semantic indexing, or LSI technique, which measures the relationships between terms in an unstructured collection of text. The index can then be used to find similar documents based on commonly occurring phrases between the documents.

   https://en.wikipedia.org/wiki/Latent_semantic_analysis

   The CSV has two columns. The first column called Question contains a natural language question that a user would ask. The second column, LinkToAnswer, contains a link to an FAQ article that provides answers to this question. Please note that this is a really small dataset created for demonstration purposes. The same question may be phrased in multiple ways in order to help the model learn multiple ways in which the question can be asked.

   Building a document vector
   We have a list of questions and related FAQ links in the data set. We first load this data set into a Panda's data frame.  we have a list of questions and related FAQ links in the data set. We first load this data set into a Panda's data frame.
   First convert all documents into lowercase. Then we remove stop words in the document using the remove stop words function in the NLTK package. We also remove the question mark character. Then we split the document into individual words and return it. Now we call the process document function for each question or document in the documents variable. This returns a document vector which gets stored in doc_vectors. 

   Creating the LSI model

   Latent semantic analysis (LSA) is a technique in natural language processing, in particular distributional semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. LSA assumes that words that are close in meaning will occur in similar pieces of text.

   LSA can use a document-term matrix which describes the occurrences of terms in documents; it is a sparse matrix whose rows correspond to terms and whose columns correspond to documents. A typical example of the weighting of the elements of the matrix is tf-idf (term frequency–inverse document frequency): the weight of an element of the matrix is proportional to the number of times the terms appear in each document, where rare terms are upweighted to reflect their relative importance.

   First we create a dictionary based on the document vectors. The dictionary is a unique list of words found in these document vectors. To do this, we use the `corpora.dictionary method`. This generates a dictionary with words and corresponding identifiers.
   Next convert the document vector into a corpus based on the identifiers in the dictionary. We use the `doc2bow` method to convert the vectors into this corpus. 
   As we can see, each word in the document is mapped to a tuple. The first number in the tuple is the word identifier in the dictionary. The second number is the total number of times this word appears in this document. Now let's build the similarity index from LSI model method found in the gensim package.

   The matrix lists the similarity code for this document with the other documents in the input. We have 10 input documents, so we get a 10 by 10 matrix. For example, the second array lists the similarity score of the second document with all other documents in this corpus. Its similarity to itself is one. The higher the similarity, the more related these documents are.

   Recommending FAQs
   First need to run this question through the same processing we did with the training dataset. We use the process document function to cleanse the question and then convert it into a corpus. Then we call the LSI method with this corpus as the index. It returns an equal and LSI model. Then we find the similarity of this model with all the other questions in our training dataset. This returns the similarity scores for this question to all other documents in the training dataset.
   The scores are a tuple, with the first number indicating the document ID and the second number the similarity score. The higher the score, the more matching is this question to the document in the dataset. To find the top matching question, we do an argsort to sort the similarity scores based on the score and return the index of the document in descending order.
   We use the question as the document for training, but we can instead use the entire content of the FAQ article as the document also. This would require a lot more processing, but can lead to more accurate results.
    