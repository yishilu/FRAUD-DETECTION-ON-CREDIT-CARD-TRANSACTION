# FRAUD-DETECTION-ON-CREDIT-CARD-TRANSACTION

Seven steps that we went through to generate the six supervised machine learning algorithms are as following: 
1.	Data cleaning, which fills in the necessary missing fields with reasonable numbers. 
2.	Variable creation and Z-scaling. 371 new variables are created and Z-scaled. 
3.	Training/testing/OOT. The entire dataset is split into training, testing and out-of-time (OOT) set. 
4.	Univariate Kolmogorov-Smirnov score (KS) and Fraud Detection Rates (FDR) calculation. For the 371 new variables, the univariate KS and univariate FDR at 3% are calculated. The 371 new variables are then sorted based on their KS and FDR at 3%. 
5.	Feature selection. The 186 new variables that have the high KS and FDR are first selected. A forward selection is then performed on the 186 variables to come up with 20 variables. 
6.	Machine learning algorithms. Six machine learning algorithms are trained by using the selected variables. The six machine learning algorithms are Logistic Regression, Neural Network, Random Forest, Boosted Trees, Support Vector Machine (SVM), and K-Nearest Neighbors (K-NN). 
7.	Machine learning algorithms evaluation. For each of the six machine learning algorithms, the effectiveness of catching credit card fraud is measured through the calculation of the FDR at 3%. 


File description:
1.  New Variables_Cardnum+Merchstate.ipynb: jupyter notebook with python code (rolling window function in Panda) to create part of the new variables for Step 2
2.  Calculate Univariate KS and FDRs for All Variables.ipynb: jupyter notebook with python code written for Step 4
3.  model.R: R code to build logistic regression, random forest, xgboost, and neural net models for Step 6
4.  FDR @ 3% for RF Model.xlsx: Tables of FDR at 1%-20% for the best model selected (Random Forest Model with 400 trees and 8 variables sampled for splitting) on training, testing and out-of-time datasets
5.  The Fraud Detection of Credit Card Transaction.pdf: detailed report
