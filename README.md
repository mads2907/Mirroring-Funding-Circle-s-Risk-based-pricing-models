# Mirroring-Funding-Circle-s-Risk-based-pricing-models
Machine and deep learning based default prediction models for SMEs. 

This repository contains, the code, data, and written Thesis for my Master Thesis project. 

I use a single classification tree, gradient boosting, bagging, random forest, Support Vector Machines, Multilayered Perceptrons, 
and Long Short-Term Memory models to predict firm specific default probability. The deep neural nets are coded in Python using 
the Keras API running on top of TensorFlow. 

The Models are trained on past financial information derived from the SMEs financial statements. The data is structured as  
panel/longitudinal and contain past financial information of 42,000 individual SMEs. 
The data contains a significant imbalance of the minority class "default". I use SMOTE+TOMEK to balance the data. 

Later, the predicted default probabilities are used in a risk-based pricing model to first assign firm-specific credit-ratings, and later
to price individual SME credit contracts, based on Funding Circle's business model. Thus, I evaluate the models on 
a non-symmetrical loss metric.  
