# EEG Brain Signals Emotion Classification

This repository focuses on the classification of emotions based on EEG brain signals. The project is an extension and improvement upon the original work found [here](https://www.kaggle.com/code/oliverright/eeg-brain-signals-emotion-classification).

## Enhancement

While the original Kaggle code provided a foundational understanding and a basic model for EEG emotion classification, this repository introduces a more advanced model: a combination of Convolutional Neural Network (CNN) and Gated Recurrent Unit (GRU) layers. This hybrid model aims to capture both spatial and temporal features in the EEG data, leading to improved classification performance.

## Model Architecture

The CNN-GRU model, named "model_2", consists of the following layers:

- Input Layer
- Reshape Layer
- Conv1D (with Batch Normalization and Max Pooling)
- Another Conv1D (with Batch Normalization and Max Pooling)
- Bidirectional GRU Layer (with Batch Normalization)
- Flatten Layer
- Dense Layers (with Dropout and Batch Normalization)

The model has a total of 20,890,211 parameters, of which 20,889,059 are trainable.

## Models Evaluated

- Gaussian Naive Bayes (GNB)
- Support Vector Machine (SVM)
- Logistic Regression (LR)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)
- Gradient Boosting
- AdaBoost
- Quadratic Discriminant Analysis (QDA)
- Nearest Centroid
- Linear Support Vector Classifier (Linear SVC)
- Kernelized Support Vector Machine (Kernelized SVM)
- Perceptron
- Linear Discriminant Analysis (LDA)
- Brain Waves GRU-CNN

## Visualization

Various visualization techniques are used to provide a clear understanding of the model performance:

1. **Bar Charts**: For a direct comparison of model performance.
2. **Heatmaps**: To visualize the correlation between different metrics.
3. **3D Scatter Plots**: For a three-dimensional comparison of precision, recall, and F1 score.
4. **Bubble Charts**: Where the x-axis represents precision, the y-axis represents recall, and the bubble size indicates the F1 score.

## Usage

1. Clone this repository.
2. Ensure you have the necessary libraries installed.
3. Run the provided Python scripts to train and evaluate the model on your EEG dataset.

## Dependencies

- Python 3.x
- Matplotlib
- Seaborn
- Other data visualization libraries as needed.

## Contribution

Feel free to fork this repository and add your own models or visualization techniques. Pull requests are welcome!

## Acknowledgment

Special thanks to the original author at Kaggle for providing the foundational code. This project aims to build upon that work and push the boundaries of EEG emotion classification.

## License

This project is licensed under the MIT License.
