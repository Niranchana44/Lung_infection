
### Project Overview



### Motivation

The project explored multiple deep learning architectures for detecting lung infections from chest images.


### Models experimented

1) A baseline convolutional neural network (CNN) was implemented with Conv2D, Batch Normalization, Dropout, and Dense layers.

Input size: 48×48

Loss: Categorical Crossentropy

Optimizer: RMSProp

Performance: Moderate accuracy, with signs of overfitting due to small dataset size.

What were the key challenges faced :
Challenge: Limited dataset size caused the model to overfit.

Due to the limited dataset, the project explored transfer learning by leveraging pretrained architectures (MobileNet and DenseNet121) trained on ImageNet, allowing the model to learn high-level features. Initially, the pretrained weights were used with frozen layers to retain general visual features. Later, selected layers were unfrozen and fine-tuned to adapt the models to lung infection classes, which differ from the original ImageNet categories.

2)  MobileNet (Transfer Learning)

MobileNet pretrained on ImageNet was used with frozen base layers, followed by custom Dense and Dropout layers.

Input size: 224×224

Loss: Categorical Crossentropy

Optimizer: RMSProp

Metrics: Accuracy, F1 Score

Performance: ~85–90% accuracy and F1 ≈ 0.81
 Performed efficiently and generalized well

 3) DenseNet121 pretrained on ImageNet was used with frozen base layers, followed by custom layers

Input size: 224×224

Loss: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall, F1 Score

EarlyStopping: Patience=6, restore_best_weights=True

Performance: 98%+ accuracy, F1 ≈ 0.98


### Tools used
Tensorflow with Keras API

