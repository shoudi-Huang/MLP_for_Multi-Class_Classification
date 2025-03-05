# Multi-Class Classification Using Multi-Layer Neural Network

## Project Overview
This project focuses on implementing a **Multi-Layer Neural Network (MLP)** from scratch to perform a **multi-class classification** task on a dataset containing 128 features and 50,000 instances, divided into 10 classes. The goal is to classify the dataset using a neural network without relying on deep learning frameworks like PyTorch or TensorFlow. Instead, the implementation uses scientific computing libraries such as NumPy and SciPy. The project explores various optimization and regularization techniques, including **ReLU activation**, **weight decay**, **momentum in SGD**, **dropout**, **batch normalization**, and **mini-batch training**, to improve the model's performance.

## Key Features
- **Multi-Layer Neural Network**: Implemented from scratch using NumPy and SciPy, with support for multiple hidden layers.
- **Optimization Techniques**: Includes **momentum in SGD**, **weight decay (L2 regularization)**, and **mini-batch training** to improve convergence and reduce overfitting.
- **Regularization Methods**: Utilizes **dropout** and **batch normalization** to enhance generalization and stability during training.
- **Activation Functions**: Implements **ReLU** and **GELU** activation functions, with ReLU being the preferred choice due to its simplicity and performance.
- **Evaluation Metrics**: Evaluates the model using **accuracy**, **precision**, **recall**, **F1-score**, and **cross-entropy loss**.
- **Hyperparameter Tuning**: Conducts extensive hyperparameter tuning to optimize the model's performance, including layer sizes, dropout rates, learning rates, and batch sizes.

## Technical Details
- **Programming Language**: Python 3
- **Preprocessing**: Normalization and standardization are applied to the dataset to improve convergence and reduce the impact of outliers.
- **Model Architecture**: The best-performing model consists of one hidden layer with 200 neurons, ReLU activation, and batch normalization.
- **Training**: The model is trained using mini-batch SGD with momentum, and cross-entropy loss is used as the loss function.

## Project Tasks
1. **Data Preprocessing**: Normalize and standardize the dataset to prepare it for training.
2. **Model Implementation**: Implement the MLP from scratch, including forward and backward propagation, activation functions, and loss computation.
3. **Optimization and Regularization**: Integrate optimization techniques like momentum and weight decay, and regularization methods like dropout and batch normalization.
4. **Hyperparameter Tuning**: Conduct hyperparameter tuning to find the best combination of layer sizes, dropout rates, learning rates, and batch sizes.
5. **Evaluation**: Evaluate the model using accuracy, precision, recall, F1-score, and cross-entropy loss. Visualize the training process and results using confusion matrices and accuracy/loss plots.

## Results and Discussion
- **Best Model Performance**: The best model achieved a **43.63% test accuracy** after 50 epochs of training. The model showed good performance in predicting certain classes (e.g., labels 0, 1, 6, 7, 8, 9) but struggled with others (e.g., labels 2, 3, 4, 5).
- **Hyperparameter Analysis**: The optimal hyperparameters included one hidden layer with 200 neurons, a dropout rate of 0.1, a learning rate of 0.001, and a batch size of 128.
- **Ablation Study**: The study revealed that **momentum in SGD** and **mini-batch training** significantly improved model performance, while **dropout** and **weight decay** had a negative impact on this dataset.
- **Comparison with Other Models**: The best model outperformed other configurations, including models with more hidden layers and advanced activation functions like GELU.

## Acknowledgments
This project was developed as part of the **COMP5329 - Deep Learning** course, focusing on the implementation of a multi-layer neural network from scratch. The goal was to gain a deeper understanding of neural network components, optimization techniques, and regularization methods, and to apply them to a real-world classification task.

## Future Work
- Explore more complex architectures, such as deeper networks or convolutional layers, to improve classification accuracy.
- Implement **grid search** for more efficient hyperparameter tuning.
- Investigate other advanced optimization techniques, such as **Adam** or **RMSprop**, to further enhance model performance.
