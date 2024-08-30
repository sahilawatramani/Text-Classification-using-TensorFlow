# Text-Classification-using-TensorFlow
# Wine Reviews Sentiment Analysis

This project aims to classify wine reviews based on their descriptions as either high-rated (90 points or above) or low-rated (below 90 points) using Natural Language Processing (NLP) techniques and neural networks. The project utilizes TensorFlow and TensorFlow Hub to create a text classification model.

#### **Dataset**

The dataset used for this project is a CSV file named `wine-reviews.csv`, which contains wine reviews along with associated metadata. The relevant columns used in this project are:

- `country`: The country of origin of the wine.
- `description`: The textual description of the wine.
- `points`: The rating of the wine on a scale of 1 to 100.
- `price`: The price of the wine.
- `variety`: The variety of the wine.
- `winery`: The winery that produced the wine.

#### **Preprocessing**

1. **Loading the Data**: The dataset is loaded, and rows with missing values in the `description` and `points` columns are removed.
   
2. **Creating Labels**: A new binary label is created where a wine is labeled as `1` if it has a rating of 90 points or above and `0` otherwise.

3. **Splitting the Dataset**: The dataset is randomly split into three parts:
   - **Training Set**: 80% of the data
   - **Validation Set**: 10% of the data
   - **Test Set**: 10% of the data

#### **Modeling**

Two models are developed:

1. **Model 1: Pretrained Embedding Layer**
   - A pre-trained word embedding layer from TensorFlow Hub (`nnlm-en-dim50/2`) is used.
   - The model consists of:
     - A dense layer with ReLU activation
     - A dropout layer with a rate of 0.4
     - Another dense layer with ReLU activation
     - A dropout layer with a rate of 0.4
     - A final dense layer with a sigmoid activation for binary classification
   - **Optimization**: The model is compiled using the Adam optimizer, with binary cross-entropy as the loss function and accuracy as the evaluation metric.

2. **Model 2: Custom Embedding with LSTM**
   - A custom text vectorization layer is used to tokenize and vectorize the text data.
   - The model consists of:
     - An embedding layer
     - An LSTM layer with 32 units
     - A dense layer with ReLU activation
     - A dropout layer with a rate of 0.4
     - A final dense layer with a sigmoid activation for binary classification
   - **Optimization**: Similar to Model 1, this model uses the Adam optimizer with binary cross-entropy as the loss function and accuracy as the evaluation metric.

#### **Training**

Each model is trained for 5 epochs, and the performance is evaluated on the validation set. After training, the model is evaluated on the test set to determine its generalization performance.

#### **Usage**

To run the project, follow these steps:

1. **Install Dependencies**: Ensure you have Python installed along with the necessary libraries.
   
   ```bash
   pip install numpy pandas matplotlib tensorflow tensorflow_hub
   ```

2. **Run the Script**: Execute the script in a Python environment to train the models and evaluate their performance.

3. **Visualize the Data**: A histogram of the wine points is plotted for initial exploratory analysis.

#### **Conclusion**

This project demonstrates how to build and evaluate text classification models using pre-trained embeddings and custom embeddings with LSTM in TensorFlow. The performance of these models can be further improved with hyperparameter tuning, additional data preprocessing, or more advanced architectures.

---

**Author**: Sahil Awatramani 

---
