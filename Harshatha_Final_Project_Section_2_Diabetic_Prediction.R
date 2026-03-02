# Classification Problem #

library(caret)            #Used for classification statistical model 
library(ggplot2)          #Used for plotting the graphs for data visualization
library(dplyr)            #For EDA, mutation
library(moments)

#Load the data-set#
print("STEP-1 : Data Pre-processing")
print("STEP-1a : Read the data-set from CSV file")
my.data <- read.csv("Harshatha_Dataset.csv")

print("STEP-1b : Sampling the data-set(first 6 records)")
head(my.data)

print("STEP-1c : Name of the features in the data-set")
colnames(my.data)

print("STEP-1d : Number of the features in the data-set")
ncol(my.data)

print("STEP-1e : Number of the data-set")
nrow(my.data)

print("STEP-2 : Exploratory Data Analysis (EDA)")
print("STEP-2a : Removing any missing/NA data from the data-set")
# Is there any missing/NA values in the data-set ? #
print(sum(is.na(my.data)))

if (sum(is.na(my.data)) == 0) {
    print("EDA-2a : NO MISSING-VALUE/NA in the given data-set")
} else {
  print("EDA-2a : Replace all MISSING-VALUE/NA with MEDIAN")
  na.len <- sum(is.na(my.data))
  print(paste("Number of NA in the data-set:", na.len))
  print("EDA-2a : After replacing all MISSING-VALUE/NA with MEDIAN")
  my.data <- my.data %>% 
    mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
  na.len <- sum(is.na(my.data))
  print(paste("Number of NA in the data-set:", na.len))
}

print("STEP-2b : FIND SKEWNESS OF THE DATA-SET")
numeric_data <- my.data %>%
  select_if(is.numeric)

skewness_results <- apply(numeric_data, 2, skewness)

barplot(skewness_results, names.arg = colnames(numeric_data),
        main = "Skewness of Columns in CSV",
        xlab = "Columns", ylab = "Skewness Value")

print("STEP-2b : Find the summary of the skewness of the data-set")
describe(my.data)

print("STEP-3 : Setting the seed for generating random number to get the same results alway ")
set.seed(100)


print("STEP-4 : Data Modeling")
print("STEP-4a : Data partitioning for training & test the data-set")
split.data <- createDataPartition(my.data$Outcome,p=0.8,list=FALSE)

#Factored since have only two outcomes - Getting the error otherwise
my.data$Outcome =as.factor(my.data$Outcome)

print("STEP-4b : Sample output from train-data")
train.data <- my.data[split.data,]
test.data <- my.data[-split.data,]
print("STEP-4b : Sample output from train-data")
print(paste("STEP-4c : Number of records in train-data", nrow(train.data)))
head(train.data,5)
print("STEP-4d : Sample output from test-data")
print(paste("STEP-4e : Number of records in train-data", nrow(test.data)))
head(test.data,5)


#Scatter plot to represent to compare Training & Test distribution#
#Scatter plot for 80% , 20% sub-net
###############################
# SVM model (polynomial kernel)
# Build Training model

print("STEP-5 : Classification model using svmPoly - Support Vector Machine with a polynomial kernel  ")
#method = "svmPoly": This argument specifies that the SVM algorithm with a 
#polynomial kernel will be used for model training. SVM is a powerful 
#supervised machine learning algorithm used for classification and regression 
#tasks, among others. The polynomial kernel is a specific type of kernel 
#function used in SVM, and it allows the algorithm to learn complex, nonlinear 
#relationships between the predictors and the outcome.

#preProcess=c("scale","center"), 
#Scaling : also known as normalization, The goal of data normalization is 
#to ensure that all features have similar ranges or distributions, which can 
#help improve the performance and convergence of various machine learning 
#algorithms.
# (1) MinMax Scaling (2) Z-score 
#Center : The centering process involves subtracting the mean value of the 
#variable from each data point within that variable. The purpose of centering 
#is to remove the average or mean effect from the data, making the data points 
#centered around zero.  
#(1) mean subtraction or mean centering,

#trControl= trainControl(method="none"),
#In this context, trControl = trainControl(method="none") explicitly sets the 
#resampling method to "none," meaning that no resampling will be performed 
#during model training.

#tuneGrid = data.frame(degree=1,scale=1,C=1)
#tuneGrid argument is used to specify a grid of hyperparameters that will 
#be used for model tuning. Model tuning involves selecting the best set of 
#hyperparameters for a machine learning algorithm to optimize its performance.


#Training Model
print("STEP-5a : Creating Classification Model")
Model <- train(Outcome ~ ., data = train.data,
               method = "svmPoly",
               na.action = na.omit,
               preProcess=c("scale","center"),
               trControl= trainControl(method="none"),
               tuneGrid = data.frame(degree=1,scale=1,C=1)
)

print("STEP-5b : Make a prediction on Training-data")
Model.training <- predict(Model, train.data) # Apply model to make prediction on Training set
print("STEP-5c : Make a prediction on Test-data")
Model.testing  <- predict(Model, test.data)  # Apply model to make prediction on Testing set

print("STEP-6 : Creating Confusion matrix to Assess the performance")
#The confusion matrix is a table that allows us to assess the performance 
#of a classification model by comparing the predicted class labels with the 
#actual class labels in the training dataset.

Model.training.confusion <-confusionMatrix(Model.training, train.data$Outcome)
Model.testing.confusion <-confusionMatrix(Model.testing, test.data$Outcome)

print("STEP-6a : Plot the confusion matrix to check efficiency of our model")
print("Confusion matrix for Training-data set")
print(Model.training.confusion)
print("Confusion matrix for Test-data set")
print(Model.testing.confusion)

print("STEP-7 : Improve the model accuracy using Cross-Validation Sampling ")
#Cross-validation is a resampling technique commonly used in machine learning
#It involves splitting the dataset into "k" subsets or folds, training 
#the model on "k-1" folds, and testing it on the remaining fold. This process 
#is repeated "k" times, with each fold being used as the test set exactly once. 

#Larger numbers of folds can provide a more accurate estimate of performance 
#but may be computationally expensive, especially with large datasets.

# Build CV model
print("STEP-7a : Create a classification model using svmPoly with CV sampling")
Model.cv <- train(Outcome ~ ., data = train.data,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv", number=20),
                  tuneGrid = data.frame(degree=1,scale=1,C=1)
)

print("STEP-7b : Make a prediction on Training-data with 20 fold")
Model.cv <-predict(Model.cv, train.data) # Perform cross-validation
print("STEP-6a : Plot the confusion matrix to check efficiency of new model")
Model.cv.confusion <-confusionMatrix(Model.cv, train.data$Outcome)
print("Confusion matrix for Training-data set")
print(Model.cv.confusion)

#Feature importance
#varImp(Model) is used to calculate the variable importance scores for the 
#features (predictor variables) in the trained model 
print("STEP-8 : Variable(Feature) Importance score")
Importance <- varImp(Model)
plot(Importance)
plot(Importance, col="red")




