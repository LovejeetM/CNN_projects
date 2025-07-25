# PyTorch CNN Projects: Anime Image Classification and Fashion MNIST Classification

This repository contains two projects that leverage Convolutional Neural Networks (CNNs) for image classification tasks using PyTorch. The first project tackles a custom binary classification problem for anime character images, while the second addresses the multi-class Fashion MNIST dataset, highlighting the role of batch normalization.

---

## 1. Anime Image Classification using a Custom CNN

### Project Overview
This project demonstrates the process of building a Convolutional Neural Network from the ground up to solve a custom, binary image classification problem. The goal is to train a model that can accurately distinguish between images of two different anime characters, 'Anastasia' and 'Takao'. This exercise covers the entire workflow, including loading a custom image dataset from a zip file, creating a custom PyTorch `Dataset` class, building a CNN architecture, and training and evaluating the model's performance.

### Dataset and Preprocessing
A custom dataset required a specialized data handling pipeline.

* **Data Source**: The dataset is provided as a `zip` archive containing 100 images in total: **50 images of 'Anastasia'** and **50 images of 'Takao'**. This represents a small but perfectly balanced dataset, ideal for a focused classification task.

* **Custom Data Handling**:
    * A custom `AnimeDataset` class was implemented, inheriting from `torch.utils.data.Dataset`. This is a crucial step for integrating custom data into PyTorch's ecosystem. The class handles the logic for loading images and their corresponding labels (0 for 'anastasia', 1 for 'takao').
    * Images were loaded directly from the zip archive into memory.

* **Image Transformations**: To prepare the images for the CNN, a series of transformations were applied using `torchvision.transforms`:
    1.  `transforms.Resize((64, 64))`: All images, regardless of their original size, were resized to a uniform 64x64 pixels. This ensures that the input tensor for the network is always a consistent shape.
    2.  `transforms.ToTensor()`: The PIL Images were converted into PyTorch Tensors. This step also scales the image pixel values from a range of [0, 255] to [0.0, 1.0].
    3.  `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`: Each of the three RGB channels of the image tensor was normalized to have a mean of 0.5 and a standard deviation of 0.5. This transforms the pixel value range to [-1.0, 1.0], which helps stabilize training.

* **Data Splitting**: An 80/20 train-validation split was created from the 100 images. Instead of splitting the data directly, the indices were split using `sklearn.model_selection.train_test_split`. PyTorch's `SubsetRandomSampler` was then used to create `DataLoader` instances that sample exclusively from either the training or validation indices, ensuring a proper and separate evaluation set.

### Methodology and Model Architecture
A standard CNN architecture was designed to learn the distinguishing features of the two characters.

* **Model (`AnimeCNN`)**: The network architecture consists of two convolutional blocks followed by two fully-connected layers:
    * **Convolutional Block 1**: A 3x3 convolutional layer (`nn.Conv2d`) with 32 filters and `padding=1` (to preserve dimensions), followed by a `ReLU` activation and a 2x2 max pooling layer (`nn.MaxPool2d`) to downsample the feature map.
    * **Convolutional Block 2**: A second 3x3 convolutional layer with 64 filters and `padding=1`, also followed by a `ReLU` activation and 2x2 max pooling.
    * **Flattening**: The resulting 64x16x16 feature map is flattened into a one-dimensional vector.
    * **Fully-Connected Layers**: A dense layer (`nn.Linear`) reduces the feature vector to 128 dimensions, followed by a final output layer that produces 2 logits for the binary classification.
* **Debugging with Forward Hooks**: A forward hook was temporarily registered on each layer to print the output tensor shape during a forward pass. This is a powerful debugging technique used to verify that the dimensions are changing as expected throughout the network, which is essential for ensuring layers connect correctly.
* **Training**: The model was trained for 5 epochs using the **Adam optimizer** (`lr=0.001`) and **Cross-Entropy Loss**, which is standard for classification.

### Results and Analysis
The model demonstrated exceptionally high performance on this specific task.

* **Performance**: The model achieved **100% accuracy** on the validation set. The validation loss dropped to nearly zero after just two epochs, indicating that the model was able to perfectly separate the 20 images in the validation set.
* **Experiment with Leaky ReLU**: A second experiment was conducted using **Leaky ReLU** instead of the standard ReLU activation function. Leaky ReLU allows a small, non-zero gradient for negative inputs, which can help prevent the "dying ReLU" problem. This model also achieved perfect performance, further confirming that the features of the two characters were easily separable for the CNN.
* **Conclusion**: The project was a success, demonstrating the ability of a simple CNN to achieve perfect classification on a small, clean dataset. However, it's important to note that with a validation set of only 20 images, this high accuracy might reflect the model's ability to memorize the specific data rather than true generalization. Testing on a larger, more diverse, and unseen dataset would be necessary to assess its real-world performance.

---

## 2. Fashion MNIST Classification using CNN with Batch Normalization

### Project Overview
This project tackles the Fashion MNIST dataset, a popular benchmark for image classification. The goal was to build a CNN to classify 10 different types of clothing and, more importantly, to demonstrate and evaluate the impact of **Batch Normalization** on the model's training process and performance.

### Dataset and Preprocessing
The project uses the well-known Fashion MNIST dataset, available directly through `torchvision`.

* **Data Source**: `torchvision.datasets.FashionMNIST`, which contains 60,000 training images and 10,000 test images.
* **Data Characteristics**: Each image is a small, 28x28 grayscale picture of a clothing item from one of 10 categories (e.g., T-shirt, Trouser, Pullover, Dress, Sneaker).
* **Image Transformations**: A simple transformation pipeline was used to resize the images to **16x16 pixels** and convert them to PyTorch tensors. Reducing the image size makes the network computationally lighter and faster to train, though at the cost of some visual detail.
* **DataLoaders**: The training and validation datasets were loaded into `DataLoader` instances with a batch size of 100.

### Methodology and Model Architecture
The core of this project is the implementation and comparison of two CNN architectures: one with batch normalization and one without.

* **Model with Batch Normalization (`CNN_batch`)**: This model was designed to showcase the effectiveness of batch normalization. Its architecture is:
    * **Convolutional Block 1**: A 5x5 convolutional layer is followed immediately by a `nn.BatchNorm2d` layer. The batch norm layer normalizes the activations across the batch of images before they are passed to the `ReLU` activation function. A max pooling layer follows.
    * **Convolutional Block 2**: This structure is repeated, with a second convolutional layer followed by `nn.BatchNorm2d`, `ReLU`, and max pooling.
    * **Fully-Connected Layer**: After flattening, the feature vector is passed to a final `nn.Linear` layer. Critically, its output is also normalized using a 1D batch norm layer (`nn.BatchNorm1d`).
* **Role of Batch Normalization**: Batch Normalization is a technique designed to stabilize and accelerate the training of deep neural networks. By normalizing the inputs to each layer, it reduces "internal covariate shift," allowing for higher learning rates, acting as a form of regularization, and making the network less sensitive to the initialization of weights.
* **Control Model (`CNN`)**: A second class, `CNN`, was defined with the exact same architecture but with all the `BatchNorm` layers removed. This serves as a control to implicitly highlight the benefits provided by the `CNN_batch` model.
* **Training**: The `CNN_batch` model was trained for 5 epochs using the **SGD optimizer** with a relatively high learning rate of **0.1** and **Cross-Entropy Loss**. Using such a high learning rate is often more feasible when batch normalization is present to stabilize the learning process.

### Results and Analysis
The training results for the model with batch normalization clearly demonstrate successful learning.

* **Training Progress**: The project includes a plot that visualizes both the training cost (loss) and validation accuracy over the 5 epochs of training. The plot shows a steady and consistent decrease in cost, paired with a corresponding steady increase in accuracy.
* **Performance**: The accuracy curve shows the model's performance climbing effectively, surpassing 80% within the 5 epochs. This demonstrates that the architecture, aided by batch normalization, is capable of learning the distinguishing features of the 10 different clothing classes.
* **Conclusion**: This project successfully illustrates the implementation of a CNN for a standard multi-class image classification task. More importantly, it highlights the practical benefits of incorporating **batch normalization** into the network architecture. The stable learning trajectory, even with a high learning rate, showcases why batch normalization has become a standard component in modern deep learning models.