

combinedFile <- 'C:/Users/admin/Downloads/Factor_Analysis/Returns.csv'
prices <- read.table(combinedFile,header = TRUE,sep = ",")
prices[,c("Date")] <- as.Date(prices[,c("Date")],"%m/%d/%Y")

prices <- prices[order(prices$Date, decreasing = TRUE),]

returns = prices

returns[-nrow(returns),-1] <- returns[-nrow(returns),-1]/returns[-1,-1]-1
returns <- returns[-nrow(returns),]  
yVars <- returns[c("AAPL","XOM","CVX","VLO","S.P","FVX")]


eig <- eigen(cov(scale(yVars)))

eValues <- eig$values
eVectors <- eig$vectors

plot(eValues,type="b")
eVectors %*% t(eVectors) 

pca1 <- as.matrix(scale(yVars)) %*% eVectors[,1]
pca2 <- as.matrix(scale(yVars)) %*% eVectors[,2]
pca3 <- as.matrix(scale(yVars)) %*% eVectors[,3]

pca <- data.frame(pca1,pca2,pca3)
cor(pca)

pcaModel <- lm(returns$GOOG ~ pca1 + pca2 + pca3)
summary(pcaModel)

plot(pcaModel$resid,returns$GOOG)
acf(pcaModel$resid)


multipleModel <- lm(returns$GOOG ~ returns$S.P + returns$FVX + returns$AAPL)
summary(multipleModel)
plot(multipleModel$resid,returns$GOOG)
acf(multipleModel$resid)

principal(cov(yVars),nfactors = 3,rotate="none")



