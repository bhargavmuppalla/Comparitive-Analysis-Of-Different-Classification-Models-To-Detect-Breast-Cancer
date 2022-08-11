
# Comparitive Analysis Of Different Classification Models To Detect Breast Cancer

A brief description of what this project does and who it's for

According to global statistics, breast cancer (BC) is one of the most frequent malignancies among women globally. Early detection of BC improves the prognosis and chances of survival by allowing patients to get timely therapeutic therapy. Patients may avoid unneeded therapies if benign tumors are classified more precisely. Motivated by this, in our work, using BC dataset we classify whether a person has malignant tumor or benign tumor using logistic regression, KNN classifier, support vector machine, Kernel SVM, Naive Bayes, Decision Tree and Random Forest Classification models and compare their performances to determine which is the best model for this problem.
## I. Introduction
The use of classification and data mining technologies to categorize data is quite successful. Particularly in the medical industry, where such procedures are commonly employed in diagnosis and decision-making. Early detection of BC improves the prognosis and chances of survival by allowing patients to get timely therapeutic therapy. Patients may avoid unneeded therapies if benign tumors are classified more precisely. As a result, accurate BC diagnosis and categorization of individuals into malignant or benign groups is a hot topic of research. Machine learning (ML) is widely regarded as the approach of choice in BC pattern classification and forecast modeling due to its unique benefits in detecting essential characteristics from complicated BC datasets


#### A. Recommended screening guidelines
Mammography. The mammography is the most essential breast cancer screening test. A mammogram is a type of X-ray that is used to examine the breast. It can identify breast cancer up to two years before you or your doctor can feel the growth.
A mammography should be done once a year for women aged 40–45 who are at an average risk of breast cancer.
Starting at the age of 30, high-risk women should receive annual mammograms and an MRI.
#### B. Some Risk Factors For Breast Cancer
Age. As women become older, their chances of developing breast cancer increase. Breast cancer is seen in about 80% of women over the age of 50.
Personal experience with breast cancer. A woman who has had breast cancer in one breast is more likely to get cancer in the other breast.
Breast cancer runs in the family. If a woman's mother, sister, or daughter had breast cancer when she was young, she has an increased chance of breast cancer (before 40). Having additional relatives who have been diagnosed with breast cancer might further increase your risk.
Genetic factors. Women with particular genetic abnormalities, such as alterations in the BRCA1 and BRCA2 genes, have a greater lifetime chance of getting breast cancer. Other gene variations may also increase the risk of breast cancer.
## II. DATA PREPARATION
For breast cancer datasets, we used the UCI Machine Learning Repository.
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
Dr. William H. Wolberg, a physician at the University Of Wisconsin Hospital in Madison, Wisconsin, produced the dataset that was utilized in this work. Dr. Wolberg used fluid samples collected from patients with solid breast masses and an easy-to-use graphical computer tool called Xcyt to analyze cytological characteristics based on a digital scan to construct the dataset. The software computes 10 features from each of the cells in the sample using a curve-fitting technique, then calculates the mean value, extreme value, and standard error of each feature for the picture, returning a 30 real-valued vector.
#### A. Attribute Information
1. ID number 2) Diagnosis (M = malignant, B = benign) 3–32)
Ten real-valued features are computed for each cell nucleus:
2. radius (mean of distances from center to points on the perimeter).
3. texture (standard deviation of gray-scale values)
4. perimeter
5. area
6. smoothness (local variation in radius lengths)
7. compactness (perimeter² / area — 1.0)
8. concavity (severity of concave portions of the contour)
9. concave points (number of concave portions of the contour)
10. symmetry
11. fractal dimension (“coastline approximation” — 1)
For each picture, the mean, standard error, and "worst" or largest (mean of the three largest values) features were computed, yielding 30 features. For example, field 3 represents Mean Radius, field 13 represents Radius SE, and field 23 represents Worst Radius.
B. Objectives
The goal of this study is to identify which traits are most useful in predicting whether a cancer is malignant or benign, as well as to look for general trends that can help with model selection and hyper parameter selection. The objective is to determine if the cancer is benign or malignant. I did this by fitting a function that can predict the discrete class of fresh input using machine learning classification algorithms.
