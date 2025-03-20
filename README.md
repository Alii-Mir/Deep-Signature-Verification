# Siamese Neural Network for Signature Verification Task

## Introduction

Signature verification is a crucial task in biometric authentication, ensuring the legitimacy of financial transactions, legal documents, and identity verification systems. The ability to distinguish between genuine and forged signatures is essential for security in various applications, including banking, forensic analysis, and access control.

Similar verification tasks include fingerprint recognition, face verification, and voice authentication, all of which aim to determine the authenticity of an individual’s identity. These tasks often involve comparing a reference sample with a newly provided sample to detect inconsistencies.

A forgery occurs when an unauthorized person attempts to replicate another individual's signature. Forgeries can be categorized into three types: random forgeries, where the forger has no knowledge of the original signature; simple forgeries, where the forger attempts to mimic the structure without extensive practice; and skilled forgeries, where the forger carefully replicates the original signature.

Several methods have been developed for signature verification. Traditional approaches rely on handcrafted features such as shape descriptors, edge detection, and statistical analysis. Machine learning techniques, including support vector machines and decision trees, have also been explored. However, deep learning-based methods, particularly convolutional neural networks, have shown superior performance by automatically learning complex features.

Siamese neural networks are well-suited for verification tasks as they learn to measure the similarity between two inputs rather than relying on predefined features. This capability makes them highly effective for distinguishing between genuine and forged signatures, particularly in scenarios with limited training data. In the below figure, you can see an overview of the general architecture of a Siamese network.

![image](https://github.com/user-attachments/assets/a2fe5b15-b4d3-42ba-b817-c7e65a903762)

Such a network consists of two identical sub-networks that process input signatures independently, extracting meaningful feature representations. The verification process follows three key steps:

- Feature Extraction: Two identical sub-networks independently process the input signature pairs through convolutional layers to generate feature embeddings, capturing essential characteristics.

- Similarity Measurement: The feature embeddings are compared using a distance metric, such as the absolute difference between the extracted feature vectors, to highlight the variations between the input signatures.

- Decision Threshold: Based on the computed distance, a threshold determines whether the signatures are genuine (similar) or forged (dissimilar).

In this study, we implemented a Siamese network for signature verification, leveraging deep learning-based feature extraction and comparison. For the purpose of direct classification, we used a fully connected classification layer as the final layer, using a Sigmoid activation function, to determine whether the signatures belong to the same individual based on the similarity score.

We trained and evaluated the model using signature samples coming from the [ICDAR](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) and [GPDS](https://www.kaggle.com/datasets/adeelajmal/gpds-1150) datasets. The model's performance is assessed using accuracy, precision, recall, F1-score, and specificity to provide a comprehensive evaluation of its effectiveness.

To learn more about Siamese neural networks, please see [here ](https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513).

# Dataset:

Please download the data from my Google Drive [here](https://drive.google.com/drive/folders/1EhgXun9FOcRl-KY43Y6Nw1Pkjij55qCa?usp=sharing). The data came from Kaggle, but it was poorly organized (There was leakage.) The dataset I have provided has addressed the issues. Pairs of forged-genuine and genuine-genuine are available in the csv files. After downloading the data, please place the folder next to the "siamese-net" folder.

The figure below shows the structure of data.
![data](https://github.com/user-attachments/assets/19572be9-c72f-4d18-bd9e-42765bc35bd1)

The dataset includes:
- Training set: 29,926 signature pairs (21,692 forged pairs). 52 persons from ICDAR and 119 persons from GPDS were randomly selected.
- Testing set: 7,979 signature pairs (5,712 forged pairs). 12 persons from ICDAR and 31 persons from GPDS were used.

The dataset split follows an 80\% training and 20\% testing ratio.

# Model:

### Loss for Siamese Network

We utilized a **Contrastive Loss Function** in our Siamese network to effectively learn the similarity and dissimilarity between image pairs. The loss is defined as ($D_E$ = Euclidean distance):

- For **genuine** pairs (label = 0), the loss = $`{D_E}^2`$.

- For **forged** pairs (label = 1), the loss = $`(max(0,~margin - D_E))^2`$.

Based on our experience, we decided to set margin to 0.5. This design ensures the network learns embeddings that bring similar pairs closer while pushing dissimilar pairs apart beyond the specified margin.

### Configuration
We set batch size to 32 and did the training for 20 epochs.

### Functions:
"SiameseDataset" class is built to load the appropriate dataset. "SiameseNetwork" is our siamese CNN network.

### Training:
For training, we used Adam optimizer and an adaptive learning rate. It was run on GPUs using CUDA.
Here you can see the training losses during epochs:

![image](https://github.com/user-attachments/assets/361075eb-930c-4ecb-9401-1cc76e982d55)

### Testing:
We use the trained model and activate the `.eval()` mode. We build the test dataset and load it. Then using a favorite threshold for contrastive loss (we used 0.1), test our model. It means for each pair, if the loss is lower than the threshold it means original pair, and for upper than threshold value we predict forgery for the pair. This threshold is adaptive and can be changed. See some examples:

![image](https://github.com/user-attachments/assets/f4796b97-0380-4538-8953-c4643e962f33)

- Predicted Eucledian Distance: 0.1126 
- Contrastive Loss: 0.1501 
- Actual Label: **Forged** 
- Predicted: **Forged**

![image](https://github.com/user-attachments/assets/b480b1ed-426c-45fb-b3cb-14b8f4704893)

- Predicted Eucledian Distance: 0.1061 
- Contrastive Loss: 0.0113 
- Actual Label: **Original** 
- Predicted: **Original**





