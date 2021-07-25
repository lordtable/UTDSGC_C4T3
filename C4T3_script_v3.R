### load the edited dataset
data<-read.csv('trainingData_ed.csv',sep=",")

### set parallel processing environment

# Required
library(doParallel)

# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6. Mine has 8 cores

# Create Cluster with desired number of cores. 
# Don't use them all! Your computer is running other processes. 
cl <- makeCluster(4)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 4 

####### Stop Cluster. After performing your tasks, stop your cluster. 
# stopCluster(cl)


########### LIBRARIES ###################
library(caret)
library(summarytools)
library(dplyr)
library(factoextra)
library(class)
library(e1071)
library(caTools)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(C50)
library(ROCR)

############# data pre-processing ###############

#view(dfSummary(data),
     #file="data.html")

df<-data #a working copy


# remove duplicated rows



df<-distinct(df)


#remove miising or NA
df<-na.omit(df)

# Converting UNIX time to DateTime
#library(lubridate)

#df$DateTime<-as_datetime(df$TIMESTAMP)

#df <- df[,c(ncol(df),1:(ncol(df)-1))]


#########  CREATE TARGET FEATURE LOC ############

## create a TARGET variable comprise of bldg#-floor#, space id-relpos

df <-cbind(df,paste(df$BUILDINGID,df$FLOOR,df$SPACEID,df$RELATIVEPOSITION), 
                stringsAsFactors=FALSE)

# Move the new attribute within the dataset
df <- df[,c(ncol(df),1:(ncol(df)-1))]

# Give the new attribute in the 1st column a new name
colnames(df)[1] <-"Loc"

# convert Loc to factor

df$Loc<-as.factor(df$Loc)

# remove from data single features of loc

df<-subset(df,select=-c(LATITUDE,LONGITUDE,FLOOR,BUILDINGID,
                                  SPACEID,RELATIVEPOSITION))


########Valid instances based on RSSI########

# RSSI ranges -104 (ext weak) to 0 dB (strongest) signal
# +100dB denotes not detected WAP

df2<-df # working copy

# make those +100dB --> -110 dB, to make RSSI more like a radius

df2[ ,5:524 ][ df2[ ,5:524 ] == 100 ] <- -110


#view(dfSummary(df2),
     #file="df2.html")

# now ranges make much more sense, especially we can see the
#strongest actual signal reception on each WAP


# 1,- may need to consider VALID instances (same location and user, similar Timestamp)

df3<-df2 # another working copy


# remove rows with all WAPs==110


#df4<-subset(df3,c(5:524)!=-110)



# 2.- may need to subset by building--avoid if possible






################## Feature selection & Eng #############

# Examine Features Variance

#nearZeroVar() with saveMetrics = TRUE returns an 
# object containing a table including: frequency ratio, 
# percentage unique, zero variance and near zero variance 

nzvMetrics <- nearZeroVar(df3, saveMetrics = TRUE)
#nzvMetrics

# Identify near zero var features: remove_cols
remove_cols<-nearZeroVar(df3,names=TRUE,freqCut=400, uniqueCut=0.3)

# get all column names from data: all_cols

all_cols<-names(df3)

#remove from df: df_NZV
df_NZV<-df3[,setdiff(all_cols,remove_cols)]

#view(dfSummary(df_NZV), file="df_NZV.html")


# remove additional non-relevant columns
df_NZV<-subset(df_NZV,select=-c(USERID,PHONEID,TIMESTAMP))

df_FNL<-df_NZV # FINAL WORKING DATASET no PCA, WPA var>>0


# Training and test split BEFORE PCA

set.seed(123)
trainSize<-round(nrow(df_FNL)*0.7) 
testSize<-nrow(df_FNL)-trainSize

training_indices<-sample(seq_len(nrow(df_FNL)),size =trainSize)

trainSet<-df_FNL[training_indices,]

testSet<-df_FNL[-training_indices,] 

y_loc_train<-trainSet$Loc
y_loc_test<-testSet$Loc


############# PCA #######################

# PCA to WPA features


# 1.- calculate PCA of training set

df_pca_train<-prcomp(trainSet[,-1],scale. = TRUE)

# Viz or obtain variance explained on train set

# plot the percentages of pca explained variance


fviz_eig(df_pca_train,addlabels = TRUE,ncp=20)

# plot the cummulative
x_pca<-seq(1,200,by=1)

df_pca_eig_metrics<-get_eig(df_pca_train) #gets pca comp var metrics


plot(x_pca,df_pca_eig_metrics$cumulative.variance.percent,type='l',
     col='red',xlab="PCA Components or Dimensions",
     ylab="Cummulative Variance Explained",lwd=3)
grid()

# 2.- apply to full dataset without split & edit

df_pca_FNL_transf<-predict(df_pca_train,newdata=df_FNL) #rotated full data

df_pca_FNL_transf2<-as.data.frame(df_pca_FNL_transf) #convert from matrix to dataframe

df_pca_FNL_transf2 <-cbind(df_pca_FNL_transf2,paste(df_FNL$Loc), 
                             stringsAsFactors=FALSE)

df_pca_FNL_transf2 <- df_pca_FNL_transf2[,c(ncol(df_pca_FNL_transf2),
                                                1:(ncol(df_pca_FNL_transf2)-1))]

colnames(df_pca_FNL_transf2)[1] <-"Loc"

df_pca_FNL_transf2$Loc<-as.factor(df_pca_FNL_transf2$Loc)

n_PCA_comp<-95 # number of PCA components to achieve 95% cumm var
n_PCA_comp<-n_PCA_comp+1

# FINAL PCA TRANSFORMED DATA, RESTRICTED TO MAIN COMPONENTS
df_FNL_pca<-subset(df_pca_FNL_transf2,select=c(1:n_PCA_comp))

#view(dfSummary(df_FNL_pca), file="df_FNL_pca.html")




# 3.- then split using same indexes as normal data

trainSet_pca<-df_FNL_pca[training_indices,]

testSet_pca<-df_FNL_pca[-training_indices,] 

y_loc_train_pca<-trainSet_pca$Loc
y_loc_test_pca<-testSet_pca$Loc



########### MODELLING #########

##### Assess baseline vs PCA default performance using kNN & C5.0 ######

dat_train<-trainSet[,-1]
dat_test<-testSet[,-1]


dat_train_pca<-trainSet_pca[,-1]
dat_test_pca<-testSet_pca[,-1]


k_val<-100 # k-neighbors

# start the clock!
ptm<-proc.time()

##run knn
classifier_knn<-knn(dat_train,dat_test,cl=y_loc_train,k=k_val)

#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T


# start the clock!
ptm<-proc.time()

##run knn for pca
classifier_knn_pca<-knn(dat_train_pca,dat_test_pca,cl=y_loc_train_pca,
                    k=k_val)

#stop the clock
T_pca<-(proc.time()-ptm)/60 # elapsed time in minutes
T_pca

## compare both performances using confusion matrix

cm_knn<-table(classifier_knn,y_loc_test) # conf matrix normal data
cm_knn_pca<-table(classifier_knn_pca,y_loc_test) # conf matrix normal data


##this function divides the correct predictions by 
# the total number of predictions that tell us how accurate
## the model is.

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

ACC<-accuracy(cm_knn)
ACC_pca<-accuracy(cm_knn_pca)


## Decision Tree testing

# on normal data

# start the clock!
ptm<-proc.time()
classifier_C50<-C5.0(trainSet[-1],trainSet$Loc)
#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T

pred_C50<-predict(classifier_C50,testSet,type="class")

ACC_C50<-100*(sum(pred_C50==testSet$Loc))/length(pred_C50)


# on PCA data

# start the clock!
ptm<-proc.time()
classifier_C50_pca<-C5.0(trainSet_pca[-1],trainSet_pca$Loc)
#stop the clock
T_pca<-(proc.time()-ptm)/60 # elapsed time in minutes
T_pca

pred_C50_pca<-predict(classifier_C50_pca,testSet_pca,type="class")

ACC_C50_pca<-100*(sum(pred_C50_pca==testSet_pca$Loc))/length(pred_C50_pca)


#################### CROSS-VAL PARAM #############

#10 fold cross validation

trControl<-trainControl(method="repeatedcv",number=10,
                         repeats=1)

############### KNN #######################

# start the clock!
ptm<-proc.time()

##run knn
classifier_CV_knn<-train(Loc~.,
                         method="knn",
                         tuneGrid=expand.grid(k=1:10),
                         trControl=trControl,
                         metric="Accuracy",
                         data=trainSet_pca)
  
  
#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T


pred_CV_knn<-predict(classifier_CV_knn,testSet_pca)

cm_CV_knn<-confusionMatrix(testSet$Loc,pred_CV_knn)


############### C5.0 #######################


c50Grid<-expand.grid(.winnow=c(TRUE,FALSE),.trials=c(1,5,10,15,20),
                     .model="tree")

# start the clock!
ptm<-proc.time()

##run C5.0
classifier_CV_C50<-train(Loc~.,
                         method="C5.0",
                         tuneGrid=c50Grid,
                         trControl=trControl,
                         metric="Accuracy",
                         data=trainSet_pca)


#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T

# start the clock!
ptm<-proc.time()

pred_CV_C50<-predict(classifier_CV_C50,testSet_pca)

#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T


cm_CV_C50<-confusionMatrix(testSet$Loc,pred_CV_C50)


################# RANDOM FOREST #################

rfGrid<-expand.grid(mtry=c(1,2,3,4,5))


# start the clock!
ptm<-proc.time()

##run RF
classifier_CV_RF<-train(Loc~.,
                         method="rf",
                         tuneGrid=rfGrid,
                         preProcess=c('scale','center'),
                         trControl=trControl,
                         metric="Accuracy",
                         data=trainSet_pca)


#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T


# start the clock!
ptm<-proc.time()

pred_CV_RF<-predict(classifier_CV_RF,testSet_pca)

#stop the clock
T<-(proc.time()-ptm)/60 # elapsed time in minutes
T


cm_CV_RF<-confusionMatrix(testSet$Loc,pred_CV_RF)




################### NAIVE BAYES ####################

# First, need to discretize independent variables for
# full dataset

# need to use dplyr cut(), n=4 for quartiles
# example PackIt> students$Income.cat2 <- cut(students$Income, breaks = 4, labels = c("Level1", "Level2", "Level3","Level4"))



# then split consistently

#trainSet_pca<-df_FNL_pca[training_indices,]

#testSet_pca<-df_FNL_pca[-training_indices,] 

#y_loc_train_pca<-trainSet_pca$Loc
#y_loc_test_pca<-testSet_pca$Loc

# apply naive Bayes






#################### COMPARIN MODELS PERFORMANCE #############################


Models_comp<-resamples(list(knn=classifier_CV_knn,
                            C50=classifier_CV_C50,
                            RF=classifier_CV_RF))

# ROC curve for a random location as example for report
# 1.- Draw a random repeatable location sample 
# for the Positive class
# set.seed(1)
# loc_1<-sample(testSet_pca$Loc,size=1,replace=TRUE)
# 
# 
# 
# prob_pred_CV_knn<-predict(classifier_CV_knn,
#                           testSet_pca,
#                           type="prob")
# 
# 
# prob_pred_CV_C50<-predict(classifier_CV_C50,
#                           testSet_pca,
#                           type="prob")
# 



# ROC curve for a random location as example for report
# 1.- use the same seed
# 2.- pick a random location out of 900+ (factor levels)
# 3.- see if can yield ROC curve according to book


###########################################

save.image("C4T3_script_v3_workspace.RData")








