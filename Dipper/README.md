# Dipper as a Paraphraser
### Make sure you have the `subset_data.csv` file in the directory
#### If you don't want to run the test data on Dipper, to save time (takes about 120 minutes), you can use the `paraphrased_results.csv` in this directory and run from the Detection Cell
1. Separating the AI generated and Human Generated texts from the test dataset
2. Running Dipper Paraphraser script for baseline results
3. Passing the results to detector models, RADAR and WILD (to get scores and labels)
5. Getting AUROC, F1, Attack Success Rate (FNR), Precision, Accuracy, Recall metrics
6. Plotting the distributions of the detector score on AI samples and Human samples
7. Plotting the ROC Curve
