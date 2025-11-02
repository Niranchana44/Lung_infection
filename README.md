
### Project Overview


### Motivation

Early and accurate detection of lung infections is crucial for preventing disease progression and improving patient outcomes. However, radiographic images are often complex and require expert interpretation, which may not always be accessible in low-resource settings.
This project aims to develop an automated deep learning model capable of identifying lung infection patterns from medical images. By leveraging Convolutional Neural Networks (CNNs) and transfer learning techniques, the model seeks to support medical professionals in diagnostic decision-making, reduce workload, and enhance screening efficiency.

### Approach

Data Exploration and Visualization: Examined image distribution across three classes — Healthy, Type 1 disease, and Type 2 disease — to understand class balance.

Preprocessing and Augmentation: Applied data augmentation (random flips, rotations, rescaling, and resizing) to increase dataset variability and improve model robustness.


### 1️⃣ Dataset Overview

The dataset consists of 251 training images and 66 test images distributed across three classes:

Healthy

Type 1 disease

Type 2 disease

### Model Architecture:

Built a custom CNN with three convolutional blocks followed by fully connected layers.

Used Glorot Normal (Xavier) initialization for stable gradient flow.

Incorporated Batch Normalization and ReLU activations for faster convergence and improved performance.

Regularization and Optimization:

Mitigated overfitting using L2 regularization, Dropout, and Early Stopping with validation monitoring to restore the best weights.

Compiled the model with categorical cross-entropy loss and the RMSprop optimizer for stable optimization.


### Models experimented




### Tools used
Tensorflow with Keras API

