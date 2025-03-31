# Siamese Neural Network for Signature Verification Task <img src="https://github.com/user-attachments/assets/9dd66a22-e8ed-47d3-b260-a17598c4b539" style="width:100px;"/>

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

## Dataset:

Please download the data from my Google Drive [contact me](mohammadalimir110@gmail.com). The data came from Kaggle, but it was poorly organized (There was leakage.) The dataset I have provided has addressed the issues. Pairs of forged-genuine and genuine-genuine are available in the csv files. After downloading the data, please place the folder next to the "siamese-net" folder.

The figure below shows the structure of data.

<img src="https://github.com/user-attachments/assets/19572be9-c72f-4d18-bd9e-42765bc35bd1" style="width:700px;"/> <br />

The dataset split follows an 80\% training and 20\% testing ratio.

- Training set: 29,926 signature pairs (21,692 forged pairs). 52 persons from ICDAR and 119 from GPDS were randomly selected.
- Testing set: 7,979 signature pairs (5,712 forged pairs). 12 persons from ICDAR and 31 persons from GPDS were used.


## Preprocessing
- Conversion to grayscale
- Resize to 105x105 pixels 
- Contrast enhancement
- Intensity matching (histogram matching)
- Transformation into tensors for neural network
![image](https://github.com/user-attachments/assets/625cfae6-102c-4f8b-9b5e-85302c1d58cb)

## Network Configuration:

![image](https://github.com/user-attachments/assets/3ef5ec0c-10e1-473c-b878-f14cf75e0cad)

Training Hyper-parameters:

- Epochs: 20
- Binary Cross-Entropy Loss
- Adam Optimizer
- Initial Learning Rate: 1e-3
- Weight Decay: 0.0005
- Batch Size: 32


### Code Functions:
`SiameseDataset` class is built to load the appropriate dataset. `SiameseNetwork` is our siamese CNN network.

### Training:
For training, we used Adam optimizer and an adaptive learning rate. It was run on GPUs using CUDA.
Here you can see the training losses during epochs:

![image](https://github.com/user-attachments/assets/32e871c6-40fb-4723-b471-46b3acdc03ab)

### Testing:

The balance between different performance metrics:

![image](https://github.com/user-attachments/assets/ca5ecebb-9a91-440d-94ac-987531d4f840)

Example:\
Threshold = 0.5

Accuracy: 0.71\
Bal. Acc: 0.59

Precision: 0.76\
Recall (Sensitivity): 0.86\
Specificity: 0.32

F1 Score: 0.81

Confusion Matrix:\
FP = 1533, FN = 801\
TP = 4911, TN = 734 <br />


**ROC Curve**:
![image](https://github.com/user-attachments/assets/9e4d6aaa-5334-4ab3-97f0-cd813c49971b)

Model output probability distribution on test dataset:

![image](https://github.com/user-attachments/assets/fa0fb12f-eed5-4ef7-a2b9-a9573b5fdde2)

**Comparison**:
![image](https://github.com/user-attachments/assets/fc0fc57f-4b6e-4ae8-84c8-b8c799804b34)


**See some examples** (Probability Threshold = 0.50):

![image](https://github.com/user-attachments/assets/302f0e2d-c892-4514-8e2c-9a1d556c1a05)
- Actual Label: **Forged** 
- Predicted: **Forged**

![image](https://github.com/user-attachments/assets/d1a4a5a6-7a23-4093-8e40-0505c71c08e7)
- Actual Label: **Original** 
- Predicted: **Forged**

![image](https://github.com/user-attachments/assets/77b9c033-983d-4b56-8859-10620e7587da)
- Actual Label: **Original** 
- Predicted: **Original**

![image](https://github.com/user-attachments/assets/52c16a0d-ef22-4067-84ec-2348529d44f2)
- Actual Label: **Forged** 
- Predicted: **Forged**

