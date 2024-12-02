# Siamese Neural Networks for Signature Verification

Siamese Neural Networks are a type of deep learning architecture designed to compare two inputs and determine their similarity. They are highly effective for tasks like signature verification, where the goal is to identify whether two signatures belong to the same individual.

## How It Works
1. **Feature Extraction**: Two identical subnetworks process the input signatures independently to generate feature embeddings.
2. **Similarity Measurement**: The feature embeddings are compared using a distance metric, such as Euclidean distance.
3. **Decision Threshold**: Based on the computed distance, a threshold determines whether the signatures are genuine (similar) or forged (dissimilar).

## Applications
- **Banking and Finance**: Verifying signatures for secure transactions and preventing fraud.
- **Forensics**: Detecting forgery in handwritten documents.
- **Authentication Systems**: Ensuring reliable identity verification in secure environments.

Siamese networks are robust, adaptable, and capable of generalizing to unseen data, making them an excellent choice for signature verification and similar comparison-based tasks.

To get to know Siamese neural networks more, please see [here ](https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513).
___
# Data:
Please download the data from my Google Drive [here](https://drive.google.com/drive/folders/1HrZL2YR8pQtkb8taLDQmIg2CbH86RpYi?usp=sharing). The data came from Kaggle, but it was poorly organized (There was leakage.) The dataset I have provided has addressed the issues. Pairs of forged-genuine and genuine-genuine are available in the csv files.

The figure below shows the structure of data.
![image](https://github.com/user-attachments/assets/cc364b9f-a3d0-493d-b662-1acb2991f65a)
