# Customer Churn Prediction — Summary Report

## Problem & Theoretical Background

Customer churn refers to customers discontinuing a company's services. For telecom providers, predicting churn is commercially valuable because acquiring a new customer costs significantly more than retaining an existing one, and retained customers generate long-term value through repeat revenue. A churn model allows a company to identify at-risk customers in advance and intervene with targeted offers or support before they leave.

Evaluating such a model requires understanding the confusion matrix and its business implications. A false negative (predicting a customer will stay when they actually churn) is the costliest error, since the company takes no retention action and silently loses the customer. A false positive (predicting churn when the customer would have stayed) is comparatively cheap, typically resulting in an unnecessary retention offer. This asymmetry means recall is generally prioritized over precision in churn modeling, since maximizing the detection of true churners matters more than avoiding occasional false alarms, even though precision becomes more important if the business has limited capacity to act on flagged customers.

The Telco Customer Churn dataset used in this project is imbalanced: roughly 73% of customers did not churn versus 27% who did. This imbalance biases models toward the majority class. SMOTE (Synthetic Minority Oversampling Technique) addresses this by generating synthetic examples of the minority (churn) class in the training set, helping models learn its patterns more effectively rather than defaulting to "no churn" predictions.

Four classification algorithms were studied: K-Nearest Neighbors (a distance-based method requiring feature scaling), Gaussian Naive Bayes (a fast probabilistic model assuming feature independence), Support Vector Machines (which find an optimal separating hyperplane and perform well on both linear and non-linear problems), and Decision Trees (which split data using interpretable decision rules).

## Observations from the Analysis

Exploratory analysis confirmed several patterns consistent with churn theory. Month-to-month contract customers churned at a much higher rate than those on one- or two-year contracts, reflecting the retention value of contractual commitment. Customers within their first 0–12 months of tenure churned the most, suggesting the early relationship period is the highest-risk window. Customers paying higher monthly charges were also more likely to churn, possibly due to price sensitivity or dissatisfaction relative to perceived value. Conversely, customers subscribed to add-on services such as Tech Support and Online Security churned less, indicating that deeper service engagement correlates with loyalty.

After preprocessing (handling missing values in TotalCharges, encoding categorical variables, scaling numeric features, and engineering features like tenure groups, service counts, and an autopay flag), the four models were trained on SMOTE-balanced data and evaluated on a held-out test set using accuracy, precision, recall, F1-score, and ROC-AUC.

Gaussian Naive Bayes trained fastest but produced a feature-independence violation, since Tenure, MonthlyCharges, and TotalCharges are clearly correlated, despite this it remained a reasonable baseline. SVM, after tuning its regularization parameter C, generally provided strong predictive performance but functioned as a black box with limited interpretability. KNN's performance depended heavily on the chosen k, tuned via F1-score across multiple values. The Decision Tree, tuned on max depth, offered the best balance of interpretability and recall, with Contract type, tenure, and Monthly Charges emerging as the most important predictive features. The experiment comparing SMOTE against `class_weight='balanced'` showed both as viable imbalance-handling strategies, with the better one selected based on recall.

## Conclusion

The analysis reinforces that contract type, tenure, and pricing are the dominant drivers of churn, and that a recall-optimized, interpretable model like the Decision Tree provides the most actionable foundation for a telecom retention strategy.
