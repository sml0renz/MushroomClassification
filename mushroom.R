##usual preamble and loading data
source("https://rfs.kvasaheim.com/stat200.R") ##just using 200.. seems to have all I may need
library(readr)
mdt <- read_csv("C:/Users/18153/Downloads/mushroom.csv")
attach(mdt)
View(mdt)

##making sure I translated right
table(class, ringType)
table(capShape)
table(capSurface)
table(capColor)
table(bruises)
table(odor)
table(gillAttachment)
table(gillSpacing)
table(gillSize)
table(gillColor)
table(stalkShape)
table(stalkRoot)
table(stalkSurfaceAR)
table(stalkSurfaceBR)
table(stalkColorAR)
table(stalkColorBR)
table(veilType) #all partial, so this may not be helpful
table(veilColor)
table(ringNumber)
table(ringType)
table(sporePrintColor)
table(population)
table(habitat)

##correlation tests? 

newClass <- ifelse(class=="edible", 1, 0)
newcapShape <- ifelse(capShape=="convex", 1, 0)
newcapSurface <- ifelse(capSurface=="scaly", 1, 0)
newcapColor <- ifelse(capColor=="brown", 1, 0)
newBruises <- ifelse(bruises=="TRUE", 1, 0)
newOdor <- ifelse(odor=="none", 1, 0)
newgillAttachment <- ifelse(gillAttachment=="free", 1, 0)
newgillSpacing <- ifelse(gillSpacing=="close", 1, 0)
newgillSize <- ifelse(gillSize=="broad", 1, 0)
newgillColor <- ifelse(gillColor=="buff", 1, 0)
newstalkShape <- ifelse(stalkShape=="tapering", 1, 0)
newstalkRoot <- ifelse(stalkRoot=="bulbous", 1, 0)
newstalksurfaceAR <- ifelse(stalkSurfaceAR=="smooth", 1, 0)
newstalksurfaceBR <- ifelse(stalkSurfaceBR=="smooth", 1, 0)
newstalkcolorAR <- ifelse(stalkColorAR=="white", 1, 0)
newstalkcolorBR <- ifelse(stalkColorBR=="white", 1, 0)
newveilColor <- ifelse(veilColor=="white", 1, 0)
newringNumber <- ifelse(ringNumber=="one", 1, 0)
newringType <- ifelse(ringType=="pendant",  1, 0)
newsporeprintColor <- ifelse(sporePrintColor=="white", 1, 0)
newPopulation <- ifelse(population=="several", 1, 0)
newHabitat <- ifelse(habitat=="grasses", 1, 0)

df <- data.frame(newClass, newcapShape, newcapSurface, newcapColor, newBruises, newOdor, newgillAttachment, newgillSpacing, newgillSize, 
                 newgillColor, newstalkShape, newstalkRoot, newstalksurfaceAR, newstalksurfaceBR, newstalkcolorAR, newstalksurfaceBR,
                 newstalkcolorAR, newstalkcolorBR, newveilColor, newringNumber, newringType, newsporeprintColor, newPopulation, 
                 newHabitat)
cor(df)
cor.test(newClass, newPopulation) ##why is stalk root weird


######## Naive Bayes########

library(e1071)
mush_train <- df[1:6499,c("newBruises", "newOdor", "newringType", "newPopulation")]
train_labels <- df[1:6499,c("newClass")]
mush_test <- df[6499:8124,c("newBruises", "newOdor", "newringType", "newPopulation")]
test_labels <- df[6499:8124,c("newClass")]
mushMod1 <- naiveBayes(mush_train, train_labels, laplace = 0)
mushpreds <- predict(mushMod1,mush_test,type="raw")
predclass <- ifelse(mushpreds[,2]>=.6,1,0)

library(gmodels)
CrossTable(test_labels, predclass)

##  SVM   ###########

mushMod2 = svm(formula = train_labels ~ .,
                 data = mush_train,
                 type = 'C-classification',
                 kernel = 'linear')

mushpreds2 <- predict(mushMod2,mush_test,type="raw")
CrossTable(test_labels, mushpreds2)

#### Logistic Regression #########

x <- 1:6499
new_samp <- sample(x)
mush_train_2 = mdt[new_samp, c("bruises", "odor", "ringType", "population")]
mush_test_2 = mdt[-new_samp, c("bruises", "odor", "ringType", "population")]
newClass[-new_samp]

mushMod4 <- glm(newClass ~ mdt$bruises + mdt$odor + mdt$ringType + mdt$population, family=binomial, data=mush_train_2)
summary(mushMod4)
mushpreds4 <- predict(mushMod4, newdata = mush_test_2, type="response")
predClass4 <- ifelse(mushpreds4 >= 0.8, 1, 0)
CrossTable(newClass[-new_samp],predClass4[-new_samp])

#### Trees #########################
library(rpart)
library(rpart.plot)
mushMod5 <- rpart(newClass ~  mdt$bruises + mdt$odor + mdt$ringType + mdt$population, method="class", data = mush_train_2)
mushpreds5 <- predict(mushMod5, newdata = mush_test_2, type="class")
rpart.plot(mushMod5)
CrossTable(newClass[-new_samp], mushpreds5[-new_samp])

?rpart

