library(MASS)
data("biopsy")

require(dplyr)
#change Class response to numeric for xgboost, also remove the ID column
biopsy <- biopsy %>%
  mutate(class = ifelse(class == "malignant",0,1)) 
mydata <- biopsy
mydata[,1] <- NULL
mydata$class <- as.integer(mydata$class)
# Load the caret package
library(caret)

# Set the seed to create reproducible train and test sets
set.seed(300)

# Create a stratified random sample to create train and test sets
# Reference the outcome variable
trainIndex   <- createDataPartition(mydata$class, p=0.75, list=FALSE, times=1)
train        <- mydata[ trainIndex, ]
test         <- mydata[-trainIndex, ]

# Create separate vectors of our outcome variable for both our train and test sets
# We'll use these to train and test our model later
train.label  <- train$class
test.label   <- test$class

#transform from data.frame to matrix because xgb.Dmatrix doesn't accept data.frame in this case (?)
train<- as.matrix(train[1:9])
test<- as.matrix(test[1:9])

# Load the Matrix package
library(Matrix)

# Create sparse matrixes and perform One-Hot Encoding to create dummy variables



dtrain <- xgb.DMatrix(data = train, label = train.label)
dtest  <- xgb.DMatrix(data = test, label = test.label)
# View the number of rows and features of each set
dim(dtrain)
dim(dtest)


# Load the XGBoost package
library(xgboost)

# Set our hyperparameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              max_depth   = 20,
              eta         = 0.1,
              gammma      = 1,
              colsample_bytree = 0.5,
              min_child_weight = 1)

set.seed(1234)

# Pass in our hyperparameteres and train the model 
system.time(xgb <- xgboost(params  = param,
                           data    = dtrain,
                           label   = train.label, 
                           nrounds = 500,
                           print_every_n = 100,
                           verbose = 1))

# Create our prediction probabilities
pred <- predict(xgb, dtest)

# Set our cutoff threshold
pred.resp <- ifelse(pred >= 0.86, 1, 0)

# Create the confusion matrix
confusionMatrix(pred.resp, test.label, positive="1")


# Get the trained model
model <- xgb.dump(xgb, with_stats=TRUE)

# Get the feature real names
names <- dimnames(dtrain)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model=xgb)[0:20] # View top 20 most important features

# Plot
xgb.plot.importance(importance_matrix)

library(ROCR)

# Use ROCR package to plot ROC Curve
xgb.pred <- prediction(pred, test.label)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     avg="threshold",
     colorize=TRUE,
     lwd=1,
     main="ROC Curve w/ Thresholds",
     print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.1))
axis(2, at=seq(0, 1, by=0.1))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")



