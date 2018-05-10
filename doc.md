# Decision Tree

Simple, yet pretty effective - ~80% accuracy, when treating image as
a flat array.

# SimpleCNN 0

Simple network with 2 convolutional layers and 3 linear perceptrons.
Trained using SGD on several (~15) epochs. Due to small batch size
(4) it was very prone to sudden drop of accuracy, probably because
large noise in sample data coming from small batch size.

# MLP

Just an experiment, maybe a baseline for other models, it gets ~95%.

# PracticalCNN

Just a different name, different layer parameters, nothing interested.

# RichCNN

Potentially little bit better, more filters in conv layers
