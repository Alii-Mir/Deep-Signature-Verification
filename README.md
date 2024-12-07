# Siamese Neural Networks for Signature Verification

Siamese Neural Networks are a type of deep learning architecture designed to compare two inputs and determine their similarity. They are highly effective for tasks like signature verification, where the goal is to identify whether two signatures belong to the same individual.

### How It Works
1. **Feature Extraction**: Two identical subnetworks process the input signatures independently to generate feature embeddings.
2. **Similarity Measurement**: The feature embeddings are compared using a distance metric, such as Euclidean distance.
3. **Decision Threshold**: Based on the computed distance, a threshold determines whether the signatures are genuine (similar) or forged (dissimilar).

![image](https://github.com/user-attachments/assets/ccf9fb15-13b5-4e9a-a286-7181fafd57c8)


### Applications
- **Banking and Finance**: Verifying signatures for secure transactions and preventing fraud.
- **Forensics**: Detecting forgery in handwritten documents.
- **Authentication Systems**: Ensuring reliable identity verification in secure environments.

Siamese networks are robust, adaptable, and capable of generalizing to unseen data, making them an excellent choice for signature verification and similar comparison-based tasks.

To get to know Siamese neural networks more, please see [here ](https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513).
___
# Dataset:
Please download the data from my Google Drive [here](https://drive.google.com/drive/folders/1HrZL2YR8pQtkb8taLDQmIg2CbH86RpYi?usp=sharing). The data came from Kaggle, but it was poorly organized (There was leakage.) The dataset I have provided has addressed the issues. Pairs of forged-genuine and genuine-genuine are available in the csv files.

The figure below shows the structure of data.
![image](https://github.com/user-attachments/assets/8e8c5c57-a815-4246-98d3-ddcac3b94ecf)

After downloading the data, please rename the folder to "data", and then place the folder next to "siamese-net" folder.
___
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





