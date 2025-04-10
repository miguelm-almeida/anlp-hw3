# Make sure you have the `subset_data.csv` file in the directory
1. Separating the AI generated and Human Generated texts from the test dataset
2. Loading QWEN as a paraphraser for baseline results
3. Passing the results to detector models, RADAR and QWEN (to get scores and labels)
5. Getting AUROC, F1, Attack Success Rate (FNR), Precision, Accuracy, Recall metrics
6. Plotting the distributions of the detector score on AI samples and Human samples
7. Plotting the ROC Curve
