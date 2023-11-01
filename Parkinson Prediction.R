
# Author: Alex Khoo Shien How
# All Rights Reserved © Alex Khoo Shien How 2023


setwd("D:\\New folder\\Documents\\NTU BCG\\NTU BCG Y2S1\\CC0007 Science and Tech for Humanity\\CC0007 Proposal 2\\Parkinson")


## Step 1: Install external package "data.table"
# install.packages("data.table")
library(data.table)
readdataset.dt <- fread("parkinson.csv", check.names = FALSE)
View(readdataset.dt)



# Check for missing values 
summary(readdataset.dt)


# Step 2: Addressing class imbalance to reduce model bias

library(ggplot2)
frequency_table <- table(readdataset.dt$status)
frequency_df_formation <- as.data.frame(frequency_table)
frequency_df_formation

ggplot(frequency_df_formation , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Presence of Parkinson") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for Parkinson dataset") +
  theme(plot.title = element_text(face="bold"))


# Result: There are 48 patients with no parkinson disease and 147 patients with parkinson disease

# Undersampling Technique
majority_class <- subset(readdataset.dt, readdataset.dt$status == 1)
minority_class <- subset(readdataset.dt, readdataset.dt$status == 0)
nrow(minority_class)
nrow(majority_class)
filtered_indices <- sample(nrow(majority_class), size = nrow(minority_class))
filtered_majority_class <- majority_class[filtered_indices,]
#View(filtered_majority_class)
new_dataset.dt <- rbind(minority_class, filtered_majority_class)
View(new_dataset.dt)


library(ggplot2)
frequency_table <- table(new_dataset.dt$status)
frequency_df_formation <- as.data.frame(frequency_table)
frequency_df_formation

ggplot(frequency_df_formation , aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  xlab("Presence of status") +
  ylab("Frequency") +
  ggtitle("Frequency Barplot for status dataset") +
  theme(plot.title = element_text(face="bold"))




# Step 3: Encode the categorical variable
new_dataset.dt$status <- as.factor(new_dataset.dt$status)


# Step 4: Splitting the dataset into Training Set and Test Set
# install.packages("caTools")
dataset.dt <- new_dataset.dt[,-1]

View(dataset.dt)


library(caTools)
split = sample.split(dataset.dt$status, SplitRatio =  0.75)
training_set = subset(dataset.dt, split == TRUE)
test_set = subset(dataset.dt, split == FALSE)


View(training_set)
View(test_set)

str(dataset.dt)


# Step 5: Feature Scaling

training_set[, 1:16 := lapply(training_set[, 1:16], FUN = function(x)scale(x))]
training_set[, 18:23 := lapply(training_set[, 18:23], FUN = function(x)scale(x))]
test_set[, 1:16 := lapply(test_set[, 1:16], FUN = function(x)scale(x))]
test_set[, 18:23 := lapply(test_set[, 18:23], FUN = function(x)scale(x))]






set.seed(42)


## Logistic Regression
class(dataset.dt$status)
  
# Fitting Logistic Regression to the Dataset (Build Logistic Regression Classifier)
classifier = glm(formula = status ~ ., family = binomial, data = training_set)

# Predicting the test results - probability of new test set
prob_pred = predict(classifier, type = "response", newdata = test_set[,-17]) #Remove the last column
prob_pred
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred

View(test_set[,17])


# Making the confusion matrix
actual_values <- test_set$status
class(actual_values)
y_pred <- as.factor(y_pred)
cm = table(actual_values, y_pred)
cm


f1_score <- F1_Score(y_pred, test_set$status, positive = "1") 
# TN = 9, FP = 3, FN = 1, TN = 11
f1_score #0.8461538

--------------------------------------------------------------------------------------------------------
## K-NN Model
set.seed(42)
library(class)
?knn

y_pred = knn(train = training_set[, -17], 
             test = test_set[, -17],
             cl = training_set$status,
             k = 5)
y_pred



# Making the confusion matrix
actual_values <- test_set$status
cm = table(actual_values, y_pred)
cm

library(MLmetrics)
f1_score <- F1_Score(y_pred, test_set$status, positive = "1") 
# TN = 11, FP = 1, FN = 2, TP = 10
f1_score #0.8695652

-----------------------------------------------------------------------------------------
## Decision Tree Classification
  
  
# install.packages("rpart")
set.seed(42)
library(rpart)
classifier <- rpart(formula = status ~ ., 
                    data = training_set, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0))

summary(classifier)

# Step 5: Plot the Maximal Tree and the results
# install.packages("rpart.plot")
library("rpart.plot")
#rpart.plot(classifier, nn = T, main = "Maximal Tree") 


# Step 6: Display the pruning sequence and 10-fold CV errors
plotcp(classifier)

# Step 7: Print out the pruning sequence and 10-fold CV errors as a table
printcp(classifier)

# Step 8: Find out the most important variable and plot the bar chart of variable importance
classifier$variable.importance



var_importance <- classifier$variable.importance
var_importance
sorted_var_importance <- var_importance[order(var_importance, decreasing = TRUE)]
sorted_var_importance
rownames <- colnames(classifier$variable.importance)
rownames

barplot(sorted_var_importance, 
        names.arg = names(classifier$variable.importance),
        xlab = "Variable Importance",
        #ylab = "Variable",
        col = "darkblue",  # Change the color as needed
        horiz = TRUE,
        las = 2,
        main = "Variable Importance Bar Chart (Maximal Tree CART Model)") 
par(mar = c(5.1,15,4.1,2.1)) # bottom, left, top right


# Step 9: Extract the optimal tree
# Compute min CVerror + 1SE in maximal tree

classifier$cptable
?which.min()
CVerror.xerror <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"]
CVerror.xerror

CVerror.std <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.std

CVerror.cap <- classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xerror"] + classifier$cptable[which.min(classifier$cptable[,"xerror"]), "xstd"]
CVerror.cap




# Step 10: Find the most optimal CP region whose CV error is just below CVerror.cap in maximal tree
i <- 1
j <- 4

while(classifier$cptable[i,j] > CVerror.cap){
  i <- i + 1
}

# Step 11: Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split
cp.opt <-  ifelse(i > 1, sqrt(classifier$cptable[i,1] * classifier$cptable[i-1,1]),1)
cp.opt


# Step 12: Get the optimal tree 
classifier2 <- prune(classifier, cp = cp.opt)
printcp(classifier2, digits = 3)
?printcp()


# Step 13: Plot the CART model and corresponding variable importance bar chart
library("rpart.plot")
rpart.plot(classifier2, nn = T, main = "Optimal CART Model.csv") 

# Step 14: Print the summary of CART model and the CART model
print(classifier2)
summary(classifier2)





# Step 15: Print the variable importance bar chart
var_importance <- classifier2$variable.importance
var_importance
sorted_var_importance <- var_importance[order(var_importance, decreasing = TRUE)]
sorted_var_importance
rownames <- colnames(classifier2$variable.importance)
rownames

barplot(sorted_var_importance, 
        names.arg = names(classifier2$variable.importance),
        xlab = "Variable Importance",
        #ylab = "Variable",
        col = "darkblue",  # Change the color as needed
        horiz = TRUE,
        las = 2,
        main = "Variable Importance Bar Chart (Optimal CART Model)") 
par(mar = c(5.1,15,4.1,2.1)) # bottom, left, top right



# Step 14: Checking prediction accuracy by making the confusion matrix table
classifier2.predict <- predict(classifier2, newdata = test_set[,-17], type = "class")
classifier2.predict

results <- data.frame(test_set, classifier2.predict)
results

cm = table(results$status, results$classifier2.predict)
cm # TN = 7, FP = 5, FN = 1, TP = 11

# Step 15: Check the f1 score of the predicted results
# install.packages("MLmetrics")
library(MLmetrics)
f1_score <- F1_Score(classifier2.predict, test_set$status, positive = "1")
f1_score #  0.7857143


# Step 16: Predict the class probability 
classifier2.predictprob <- predict(classifier2, newdata = test_set[,-17], type = "prob")
classifier2.predictprob



# Random Forest Model
set.seed(42)

library(randomForest)

classifier3 = randomForest(x = training_set[,-17], y = training_set$status, ntree = 30, keep.forest = TRUE) #Note ntree large will cause overfitting

# Step 2: Predict the test results - probability of new test set
classifier3.predict = predict(classifier3, newdata = test_set[,-17]) 
classifier3.predict


# Step 3: Predict the class probability
classifier3.predict_prob = predict(classifier3, newdata = test_set[,-17], type = "prob") 
classifier3.predict_prob

# Step 4: Plot confusion matrix
cm = table(test_set$status, classifier3.predict)
cm #TN = 11, FP = 1, FN = 2, TP = 10



# Step 5: Plot a variable importance plot
varImpPlot(classifier3, main = "Variable Importance Plot of Random Forest")


library(MLmetrics)
f1_score <- F1_Score(classifier3.predict, test_set$status, positive = "1")
f1_score # 0.8695652

