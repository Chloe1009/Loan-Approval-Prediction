#Import library
library(Amelia)
library(plyr)
library(dplyr)
library(naniar)
library(ggplot2)
library(psych)
library(caret)
library(e1071)
library(VIM)
library(mice)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(ROCR)

#Get workplace (Make sure the dataset already put in the workplace)
getwd()

#Import Dataset
data <-read.csv('Loan_Approval_Dataset.csv',header = TRUE,stringsAsFactors = TRUE)

#Analizing data
head(data)
summary(data)
str(data)

#Checking and visualize missing value #na.string replace the "" and "NA" inside the data into NA
missing.data <- read.csv('Loan_Approval_Dataset.csv' , na.strings = c("", "NA") , 
                         header = TRUE , stringsAsFactors =  TRUE)
summary(missing.data)
sum(is.na(missing.data))
miss_var_summary(missing.data)
miss_plot <- aggr(missing.data, col=c('yellow','blue'),
                  labels=names(missing.data), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern") ,numbers=TRUE, sortVars=TRUE,)

#Fixing missing values with mice function 
my_imp = mice(missing.data , m = 2 , method = 'cart', maxit = 2)

#Checking missing value
my_imp$imp$Gender
my_imp$imp$Credit_History

cleandata <- complete(my_imp , 2)
summary(cleandata)
miss_var_summary(cleandata)
missmap(cleandata)
################################################################################
################ Data Visualization (Numerical Variable) #######################
############################# LoanAmount #######################################
h<-hist(cleandata$LoanAmount, 
        main="Histogram for Loan Amount", 
        xlab="Loan Amount", 
        border="red", 
        col="yellow",
        las=1, 
        breaks=50, prob = TRUE)
d <- density(cleandata$LoanAmount)
polygon(d, border="black" , lwd = 3)

############################ Applicant Income ##################################
hist(cleandata$ApplicantIncome, 
     main="Histogram for Applicant Income", 
     xlab="Income", 
     border="red", 
     col="yellow",
     las=1, 
     breaks=100, prob = TRUE)
d1 <- density(cleandata$ApplicantIncome)
polygon(d1, border="black" , lwd = 3)

########################### Co-applicant Income ################################
hist(cleandata$CoapplicantIncome, 
     main="Histogram for Co-applicant Income", 
     xlab="Income", 
     border="red", 
     col="yellow",
     las=1, 
     breaks=100, prob = TRUE)
d1 <- density(cleandata$CoapplicantIncome)
polygon(d1, border="black" , lwd = 3)

########################### Loan amount term ###################################
hist(cleandata$Loan_Amount_Term, 
     main="Histogram for Loan Amount Term", 
     xlab="Income", 
     border="red", 
     col="yellow",
     las=1, 
     breaks=100, prob = TRUE)

########################### Credit History #####################################
hist(cleandata$Credit_History, 
     main="Histogram for Credit History", 
     xlab="Income", 
     border="red", 
     col="yellow",
     las=1, 
     breaks=100, prob = TRUE)

################################################################################
######################### Data after log function ##############################
################################################################################
###################### Histogram for Log Loan Amount ###########################
cleandata$logLoanAmount <- log(cleandata$LoanAmount)
hist(cleandata$logLoanAmount  ,  main="Histogram for Log Loan Amount", 
     xlab="Log Loan Amount", 
     border="red", las=1,
     col="yellow",breaks = 30 , freq = FALSE)
d2 <- density(cleandata$logLoanAmount)
polygon(d2, border="black" , lwd = 5)

###################### Histogram for Log Total Income ##########################
cleandata$totalincome <- cleandata$ApplicantIncome + cleandata$CoapplicantIncome
cleandata$logtotalIncome <- log(cleandata$totalincome)
hist(cleandata$logtotalIncome,  main="Histogram for Log Total Income", 
     xlab="Log Total Income", 
     border="red", las = 1,
     col="yellow",breaks = 30 , freq = FALSE)
d3 <- density(cleandata$logtotalIncome)
polygon(d3, border="black" , lwd = 4)

################ Data visualization (Categorial Variables)######################
#bar plot to see correlation between each categorial variables with Loan status
par(mfrow=c(2,4))
counts <- table(cleandata$Loan_Status, cleandata$Gender)
barplot(counts, main="Loan Status by Gender",
        xlab="Gender", col=c("darkblue","red") , beside = TRUE)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue"), 
       cex = 0.7
)

counts2 <- table(cleandata$Loan_Status, cleandata$Education)
barplot(counts2, main="Loan Status by Education",
        xlab="Number of Education", col=c("darkblue","red"), beside = TRUE)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue")
)
counts3 <- table(cleandata$Loan_Status, cleandata$Married)
barplot(counts3, main="Loan Status by Married",
        xlab="Married", col=c("darkblue","red"), beside = TRUE
)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue")
)
counts4 <- table(cleandata$Loan_Status, cleandata$Self_Employed)
barplot(counts4, main="Loan Status by Self Employed",
        xlab="Self_Employed", col=c("darkblue","red"), beside = TRUE
)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue")
)
counts5 <- table(cleandata$Loan_Status, cleandata$Property_Area)
barplot(counts5, main="Loan Status by Property_Area",
        xlab="Property_Area", col=c("darkblue","red"), beside = TRUE
)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue"), 
       cex = 0.6
)
counts6 <- table(cleandata$Loan_Status, cleandata$Credit_History)
barplot(counts6, main="Loan Status by Credit_History",
        xlab="Credit_History", col=c("darkblue","red"), beside = TRUE
)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue")
)

counts7 <- table(cleandata$Loan_Status, cleandata$Dependents)
barplot(counts7, main="Loan Status by Dependent",
        xlab="Dependents", col=c("darkblue","red"), beside = TRUE
)
legend("right",
       c("Yes", "No"),
       fill = c("red","darkblue")
)

################################################################################
######################### Building the Models ##################################
################################################################################
######################### Decision Tree Model ##################################
#Split data into train set and test set for decision tree model
set.seed(5)
sample <- sample.int(n = nrow(cleandata), size = floor(.70*nrow(cleandata)), replace = F)
traindt <- cleandata[sample, ]
testdt  <- cleandata[-sample, ]

#Train, Plot and Predict the model
decisiontree <- rpart(Loan_Status ~ Gender+Married+Dependents+Education+Self_Employed
                      +ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term
                      +Credit_History+Property_Area+totalincome+logtotalIncome+logLoanAmount,
                       data = traindt, method = "class")
fancyRpartPlot(decisiontree, cex=1, type = 2, main = "Decision Tree")
DTpred <- predict(decisiontree, testdt, type = "class")
confusionMatrix(DTpred , testdt$Loan_Status)

#Prune, Plot and Predict the model with prune function from 1st model
printcp(decisiontree)
plotcp(decisiontree)
decisiontree.CP <- decisiontree$cptable[which.min(decisiontree$cptable[,"xerror"]),"CP"]
decisiontree.PRUNED <- prune(decisiontree, cp = decisiontree.CP)
fancyRpartPlot(decisiontree.PRUNED, type = 2, main = "Pruned Decision Tree")
decisiontree.PRUNED.PRED <- predict(decisiontree.PRUNED, testdt, type = "class")
confusionMatrix(decisiontree.PRUNED.PRED, testdt$Loan_Status)

################################################################################
######################### Random Forest Model ##################################
#Split data into train set and test set for random forest model
set.seed(42)
sample<-sample.int(n = nrow(cleandata), size = floor(.70*nrow(cleandata)), replace = F)
trainrf<-cleandata[sample,]
testrf<-cleandata[-sample,]
RFM <- randomForest(Loan_Status ~.-(Loan_ID)
                    , data=trainrf, ntree= 500,
                    importance=TRUE)
RFM
RFM.pred <- predict(RFM, testrf)
confusionMatrix(RFM.pred , testrf$Loan_Status)

#Finding the more important variable
varImpPlot(RFM)

#Changing to more important feature
fit_RFM <- randomForest(Loan_Status ~ Credit_History +logtotalIncome+logLoanAmount,
                        data = trainrf, ntree = 500 , importance = TRUE)

fit_RFM
RFM.newpred <- predict(fit_RFM, testrf)
confusionMatrix(RFM.newpred,testrf$Loan_Status)

################################################################################
###################### Logistic Regression Model ###############################
#Split data into train set and test set for logistic regression model
set.seed(3456)
sample<-sample.int(n = nrow(cleandata), size = floor(.70*nrow(cleandata)), replace = F)
trainlr<-cleandata[sample,]
testlr<-cleandata[-sample,]
log.reg<-glm(Loan_Status ~(Gender+Married+Dependents+Education+Self_Employed
                           +ApplicantIncome+CoapplicantIncome+LoanAmount+Loan_Amount_Term
                           +Credit_History+Property_Area+totalincome+logtotalIncome+logLoanAmount), 
             family=binomial(link='logit'),data=trainlr)
summary(log.reg)

#(refix the model with more significant features )
log.reg.rev<-glm(Loan_Status~ (Married+Dependents+Credit_History+Property_Area),
                 family=binomial(link='logit'),data=trainlr)
summary(log.reg.rev)

#the most significant feature is credit history
#people with credit history more likely to default the loan
#graph freq (x) & prob (y)
log.reg.CH<-glm(Loan_Status~ Credit_History,
                family=binomial(link='logit'),data=trainlr)
summary(log.reg.CH)
newdata<-data.frame(Credit_History=seq(min(trainlr$Credit_History),max(trainlr$Credit_History),len=500))
newdata$prob=predict(log.reg.CH,newdata,type="response")

#Change Y/N to 1/0
cleandata$Loan_Status<-ifelse(data$Loan_Status=="Y",1,0)
cleandata

#Plotting graph
adata<-data.frame(Credit_History=c(0,1))
probabilities<-log.reg.CH%>%predict(adata,type="response")
probabilities.classes<-ifelse(probabilities>0.5,"Y","N")
probabilities.classes
trainlr%>%
        mutate(prob = ifelse(Loan_Status == "Y", 1, 0)) %>%
        ggplot(aes(Credit_History,prob))+
        geom_point(alpha=0.2)+
        geom_smooth(method="glm",method.args=list(family="binomial"))+
        labs(
                title="Logistic Regression Model",
                x="Credit History",
                y="Probability of loan being approved"
        )

################
predicted.loan<-predict(object=log.reg.rev,newdata=testlr,type="response")

head(predicted.loan,n=30)

binary_predict<-as.factor(ifelse(predicted.loan>0.5,1,0))
head(binary_predict,n=30)

testlr$Loanstatusfactor<-as.factor(ifelse(testlr$Loan_Status=="Y",1,0))
confusionMatrix(data=binary_predict,reference = testlr$Loanstatusfactor)

pred_ROCR<-prediction(predicted.loan,testlr$Loanstatusfactor)
auc_ROCR <- performance(pred_ROCR, measure = 'auc')
plot(performance(pred_ROCR, measure = 'tpr', x.measure = 'fpr'), colorize = TRUE,
     print.cutoffs.at = seq(0, 1, 0.1), text.adj = c(-0.2, 1.7))
abline(a=0, b=1, col="#8AB63F")
paste('Area under Curve :', signif(auc_ROCR@y.values[[1]]))

################################################################################
############################### The End  #######################################
################################################################################

