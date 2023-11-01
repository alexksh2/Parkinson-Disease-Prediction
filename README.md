# Predictive Diagnosis of Parkinson Disease 


__Parkinson’s disease (PD)__ is a neurodegenerative disease that results in uncontrollable movements and behavioural changes (National Institute on Aging 2022). Over 1 in 4 people are misdiagnosed with a different condition, with nearly 50% given treatment for their incorrectly-diagnosed condition, resulting in over 34% reporting worse health than before as a result (Media, P. A. 2019).


In fact, research studies has indicated that __the characterisation of prediagnosis Parkinson's Disease__ (PD) and __the early prediction of the disease's progression__ are essential for preventive interventions, risk stratification and understanding of the disease pathology (Yuan, et al. 2021)

Therefore, several machine learning classification models such as __Logistic Regression, K-Nearest Neighbors Algorithm, Classification and Regression Tree Model and Random Forest__ have been modelled using the input dataset such that accurate predictive diagnosis of Parkinson Disease can be conducted using Artificial Intelligence.

![Photo](https://github.com/alexksh2/Parkinson-Disease-Prediction/assets/138288828/2489a0c0-84ce-4a1e-9d5f-9b6f26fa1249)


# Information About Dataset Used

Source of the dataset: https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set/code

__This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD)__. Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). 

# Attribute Information
1. name : ASCII subject name and recording number
2. MDVP:Fo(Hz) : Average vocal fundamental frequency
3. MDVP:Fhi(Hz) : Maximum vocal fundamental frequency
4. MDVP:Flo(Hz) : Minimum vocal fundamental frequency
5. MDVP:Jitter(%) , MDVP:Jitter(Abs) , MDVP:RAP , MDVP:PPQ , Jitter:DDP : Several measures of variation in fundamental frequency
6. MDVP:Shimmer , MDVP:Shimmer(dB) , Shimmer:APQ3 , Shimmer:APQ5 , MDVP:APQ , Shimmer:DDA : Several measures of variation in amplitude
7. NHR , HNR : Two measures of ratio of noise to tonal components in the voice
8. status : Health status of the subject (one) - Parkinson's, (zero) - healthy
9. RPDE , D2 : Two nonlinear dynamical complexity measures
10. DFA : Signal fractal scaling exponent
11. spread1 , spread2 , PPE : Three nonlinear measures of fundamental frequency variation



# Model Results
Overall Accuracy Rate : The overall accuracy of machine learning model = (TP + TN)/ (TP + TN + FP + FN) <Br>
Precision : How often a machine learning model correctly predicts the positive class  = TP / (TP + FP) <br>
Recall : The ability of a model to find all the relevant cases within a data set = TP / (TP + FN) <br>
F1-score : A machine learning evaluation metric that measures a model's accuracy by combining the precision and recall scores of a model. <br>


__1. Logistic Regression:__ <br>
   <br> F1-score = 0.8461538
   <br> Overall Accuracy Rate = 20 / 24 
   <br> False Positive = 3
   <br> False Negative = 1 <br>

__2. K-Nearest Neighbors Algorithm:__ <br>
   <br> F1-score = 0.8695652
   <br> Overall Accuracy Rate = 21 / 24 
   <br> False Positive = 1
   <br> False Negative = 2 <br>
   
__3. Classification and Regression Tree Model:__ <br>
   <br> F1-score = 0.7857143
   <br> Overall Accuracy Rate = 18 / 24 
   <br> False Positive = 5
   <br> False Negative = 1 <br>

__4. Random Forest:__ <br>
   <br> F1-score 0.8695652 
   <br> Overall Accuracy Rate = 21 / 24 
   <br> False Positive = 1 
   <br> False Negative = 2 <br>

# Conclusion:
__K-Nearest Neighbours Algorithm and Random Forest__ are the two best-performing classification models for this analysis. __However, random seed values are observed to have a significant effect on all machine learning models performance.__ __Therefore, it may be concluded that more data records are required to provide a more comprehensive assessment on all machine learning models



__All Rights Reserved © Alex Khoo Shien How 2023__

# Citations

__Media, P. A. (2019, December 30). Quarter of Parkinson’s sufferers were wrongly diagnosed, says charity. The Guardian.__ https://www.theguardian.com/society/2019/dec/30/quarter-of-parkinsons-sufferers-were-wrongly-diagnosed-says-charity

__Yuan, W., Beaulieu-Jones, B., Krolewski, R., Palmer, N., Veyrat-Follet, C., Frau, F., Cohen, C., Bozzi, S., Cogswell, M., Kumar, D., Coulouvrat, C., Leroy, B., Fischer, T. Z., Sardi, S. P., Chandross, K. J., Rubin, L. L., Wills, A.-M., Kohane, I., & Lipnick, S. L. (2021). Accelerating diagnosis of Parkinson’s disease through risk prediction. BMC Neurology, 21(1). https://doi.org/10.1186/s12883-021-02226-4Yuan, W., Beaulieu-Jones, B., Krolewski, R., Palmer, N., Veyrat-Follet, C., Frau, F., Cohen, C., Bozzi, S., Cogswell, M., Kumar, D., Coulouvrat, C., Leroy, B., Fischer, T. Z., Sardi, S. P., Chandross, K. J., Rubin, L. L., Wills, A.-M., Kohane, I., & Lipnick, S. L. (2021). Accelerating diagnosis of Parkinson’s disease through risk prediction. BMC Neurology, 21(1).__ https://doi.org/10.1186/s12883-021-02226-4 <br>

__National Institute on Aging. (2022, April 14). Parkinson’s disease: Causes, Symptoms, and Treatments. National Institute on Aging.__ https://www.nia.nih.gov/health/parkinsons-disease

