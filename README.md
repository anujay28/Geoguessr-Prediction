# Geospatial Classification using Convolutional Neural Networks
## How to Run

To run this project, follow the steps below:
1. **Clone the Repository**: Clone this repository to your local machine using the following command:
2. **Download the Dataset**: As mentioned in the code comments, the dataset can be obtained from Kaggle - ["Google Street View" dataset](https://www.kaggle.com/datasets/paulchambaz/google-street-view). Download the dataset and place it in the appropriate location within the cloned repository.
3. **Unzip the Dataset**: Unzip and include the ​```edited_data.csv​``` inside the  ​```archive/dataset/​``` directory
4. **Install Dependencies**: Ensure you have Python 3.x installed on your system. Install the required dependencies using the following command:
​```pip install torch torchvision pandas matplotlib seaborn
​```

6. **Run the Jupyter Notebook**: Open the Jupyter Notebook (`optimized_street_view.ipynb`) in your preferred Jupyter environment (Jupyter Notebook, JupyterLab, etc.). Make sure to execute the cells sequentially to ensure all necessary libraries are imported and data is loaded.
7. **Notebook Execution**: The notebook consists of cells that can be executed one after another. To execute a cell, press `Shift + Enter`. Make sure to execute all cells in order to ensure a smooth execution of the notebook.
8. **Review Results**: After the notebook execution is complete, review the model's performance, exploration, and analysis results presented in the notebook.
9. **Optional: Hyperparameter Tuning**: If you wish to experiment with different hyperparameters, feel free to modify



## Problem And Objective

The primary objective of this project is to create a machine learning model capable of accurately predicting the continent based on specific features extracted from street-view images. The inspiration behind this endeavor was to develop a powerful tool to excel in the widely popular geography guessing game known as "GeoGuessr." By successfully classifying street-view images into different continents, players can gain a competitive advantage in the game, enhancing its overall enjoyment and challenge.

However, beyond the immediate application for GeoGuessr, this project lays the foundation for broader and more robust geospatial classifications. The ultimate goal is to extend the model's capabilities to classify images at the city level, landmark level, street level, and eventually achieve highly precise latitude and longitude coordinates.

## The Data

**Data Collection**: The primary dataset was obtained from Kaggle, specifically the ["Google Street View" dataset](https://www.kaggle.com/datasets/paulchambaz/google-street-view) curated by Paul Chambaz. It consists of street-view images captured from various continents with their respective latitudes and longitudes. The code to generate these images is available on GitHub at [https://github.com/paulchambaz/geotrouvetout](https://github.com/paulchambaz/geotrouvetout).

## Tools Used

- **PyTorch**: PyTorch is a popular deep learning framework used for building and training neural networks. It provides a flexible and efficient platform for implementing various machine learning models, including Convolutional Neural Networks (CNNs) for image classification.

- **Geopy**: Geopy is a Python library that allows geocoding and reverse geocoding, enabling the retrieval of geographic information based on latitude and longitude coordinates. In this project, geopy.geocoders was used to obtain continent labels from geographical coordinates.

- **Scikit-learn**: Scikit-learn is a comprehensive machine learning library in Python. It offers a wide range of tools for data preprocessing, model selection, evaluation metrics, and data visualization. Scikit-learn was likely used for tasks such as splitting the dataset into training and testing sets, hyperparameter tuning, and evaluating model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

- **Torchvision**: Torchvision is a part of PyTorch and is specifically used for computer vision tasks. It provides pre-trained CNN models and data augmentation techniques to enhance the model's generalization ability and performance.

- **Pandas**: Pandas is a versatile library for data manipulation and analysis. It is often used to handle tabular data and perform operations like filtering, merging, and transforming data. In this project, Pandas has been used for managing datasets and extracting features.

## Model Structure

The use of Convolutional Neural Networks (CNNs) for image classification tasks is a common and effective choice, and there are several reasons why CNNs are well-suited for this task:

1. **Local Feature Extraction**: CNNs are designed to automatically learn local features from images. The convolutional layers in CNNs use small filters (kernels) to convolve over the input images, which helps them capture local patterns and features like edges, textures, and shapes.

2. **Translation Invariance**: CNNs are capable of learning translation-invariant features, which means they can recognize the same pattern or object in different parts of the image. This property is crucial for image classification, where the position of objects in the image might vary.

3. **Hierarchical Representation**: CNNs have multiple layers that learn hierarchical representations of the input images. The lower layers capture basic features, and as we move higher up in the network, the learned features become more complex and abstract. This hierarchical representation allows the model to understand the image at different levels of granularity, which is essential for classification tasks.

4. **Robust to Variations**: CNNs are robust to small variations and distortions in the input images, such as rotation, scaling, or occlusions. This robustness is essential when dealing with real-world images that may have variations in lighting, viewpoint, or background.

5. **Non-linearity**: CNNs incorporate non-linear activation functions, such as ReLU (Rectified Linear Unit), which enables them to capture complex and non-linear relationships between image features.

## Exploration and Analysis

### Model 1: Basic CNN
- Architecture: 2 Convolutional layers, 2 Fully Connected layers, and Dropout for regularization.
- Hyperparameters: Learning rate (LR) = 0.005, Momentum = 0.9, Epochs = 50.
- Results: Training accuracy was very high (99.99%), but validation and test accuracy were lower (around 60%).
- Observation: The model suffered from overfitting as indicated by the significant difference between training and validation/test accuracy.

### Model 2: Basic CNN with Reduced Momentum
- Architecture: Same as Model 1.
- Hyperparameters: Learning rate (LR) = 0.005, Reduced Momentum = 0.5, Epochs = 25.
- Results: Training accuracy decreased to ~50%, but the model did not overfit.
- Observation: By reducing the momentum, the model's capacity to overfit was reduced, but it sacrificed some accuracy.

### Model 3: CNN with Reduced Model Complexity
- Architecture: 2 Convolutional layers, 1 Fully Connected layer (reduced neurons), and increased dropout, 5-Fold Cross Validation.
- Hyperparameters: Learning rate (LR) = 0.005, Momentum = 0.75, Epochs = 30.
- Results: Training accuracy was around 71%, and validation accuracy was approximately 53%.
- Observation: This model achieved a good balance between training and validation accuracy, avoiding overfitting.

### Hyperparameters:
- Optimizer: SGD
- Loss Function: CrossEntropyLoss
- Learning Rate: 0.005 (Used in all three models)
- Batch Size: 32 (Used in all three models)
- Number of Epochs:
  - Model 1: 50
  - Model 2: 25
  - Model 3: 20
- Number of Classes: 6 (Used in all three models)
- Number of Workers: 2 (Used in all three models)
- Momentum:
  - Model 1: 0.9
  - Model 2: 0.75
  - Model 3: 0.75
- Dropout Rate:
  - Model 1: 0.25
  - Model 2: 0.25
  - Model 3: 0.5

## General Insights

- The basic CNN model without regularization showed significant overfitting, leading to high training accuracy but poor generalization to unseen data.
- By reducing the momentum in the optimization algorithm, the model's capacity to overfit was reduced, but at the cost of overall accuracy.
- In the third model, reducing the model's complexity and increasing dropout helped achieve a more generalized model, resulting in better performance on validation data.

---
