# Assignment 1 (original R submission, Nov 2024)
# GRU (20/15/10 units) projection of Swiss male mortality, trained on 1950-1999.
# Adapted from the course code accompanying Wuthrich & Merz (2023),
# "Statistical Foundations of Actuarial Learning and its Applications", Springer.
# Only file paths were changed for this repository.

options(keras.view_metrics = FALSE)
require(data.table)
require(dplyr)
require(ggplot2)
require(data.table)
require(reshape2)
require(HMDHFDplus)
require(gnm)
require(stringr)
require(ggpubr)
require(keras)
# install.packages("reticulate")
library(reticulate)
# use_python("<path to a Python env with TensorFlow>", required = TRUE)
set.seed(1234)
all_mort <- fread("../data/mortality_CHE.csv")                     
all_mort$Gender <- as.factor(all_mort$Gender)
# gru layers
GRU3 <- function(T0, tau0, tau1, tau2, tau3, y0=0, optimizer){
  Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
  Output = Input %>%  
    layer_gru(units=tau1, activation='tanh', recurrent_activation='tanh', 
              return_sequences=TRUE, name='GRU1') %>%
    layer_gru(units=tau2, activation='tanh', recurrent_activation='tanh', 
              return_sequences=TRUE, name='GRU2') %>%           
    layer_gru(units=tau3, activation='tanh', recurrent_activation='tanh', name='GRu3') %>%           
    layer_dense(units=1, activation=k_exp, name="Output",
                weights=list(array(0,dim=c(tau3,1)), array(log(y0),dim=c(1))))  
  model <- keras_model(inputs = list(Input), outputs = c(Output))
  model %>% compile(loss = 'mean_squared_error', optimizer = optimizer)
}
### prepare data for Recurrent Neural Networks 
data.preprocessing.RNNs <- function(data.raw, gender, T0, tau0, ObsYear=1999){    
  mort_rates <- data.raw[which(data.raw$Gender==gender), c("Year", "Age", "logmx")] 
  mort_rates <- dcast(mort_rates, Year ~ Age, value.var="logmx")
  # selecting data
  train.rates <- as.matrix(mort_rates[which(mort_rates$Year <= ObsYear),])
  # adding padding at the border
  (delta0 <- (tau0-1)/2)
  if (delta0>0){for (i in 1:delta0){
    train.rates <- as.matrix(cbind(train.rates[,1], train.rates[,2], train.rates[,-1], train.rates[,ncol(train.rates)]))
  }}
  train.rates <- train.rates[,-1]
  (t1 <- nrow(train.rates)-(T0-1)-1)
  (a1 <- ncol(train.rates)-(tau0-1)) 
  (n.train <- t1 * a1) # number of training samples
  xt.train <- array(NA, c(n.train, T0, tau0))
  YT.train <- array(NA, c(n.train))
  for (t0 in (1:t1)){
    for (a0 in (1:a1)){
      xt.train[(t0-1)*a1+a0,,] <- train.rates[t0:(t0+T0-1), a0:(a0+tau0-1)]
      YT.train[(t0-1)*a1+a0] <-   train.rates[t0+T0, a0+delta0]
    }}
  list(xt.train, YT.train)
}

### recursive prediction
recursive.prediction <- function(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model.p){       
  single.years <- array(NA, c(2016-ObsYear))
  
  for (ObsYear1 in ((ObsYear+1):2016)){
    data2 <- data.preprocessing.RNNs(all_mort2[which(all_mort2$Year >= (ObsYear1-10)),], gender, T0, tau0, ObsYear1)
    # MinMaxScaler (with minimum and maximum from above)
    x.vali <- array(2*(data2[[1]]-x.min)/(x.min-x.max)-1, dim(data2[[1]]))
    y.vali <- -data2[[2]]
    Yhat.vali2 <- exp(-as.vector(model.p %>% predict(x.vali)))
    single.years[ObsYear1-ObsYear] <- round(10^4*mean((Yhat.vali2-exp(-y.vali))^2),4)
    predicted <- all_mort2[which(all_mort2$Year==ObsYear1),]
    keep <- all_mort2[which(all_mort2$Year!=ObsYear1),]
    predicted$logmx <- -as.vector(model %>% predict(x.vali))
    predicted$mx <- exp(predicted$logmx)
    all_mort2 <- rbind(keep,predicted)
    all_mort2 <- all_mort2[order(Gender, Year, Age),]
  }
  list(all_mort2, single.years)
}                  

##### dataset
T0 <- 10
tau0 <- 5
gender <- "Male"
ObsYear <- 1999
# training data pre-processing 
data1 <- data.preprocessing.RNNs(all_mort, gender, T0, tau0, ObsYear)

# validation data pre-processing
all_mort2 <- all_mort[which((all_mort$Year > (ObsYear-10))&(Gender==gender)),]
all_mortV <- all_mort2
vali.Y <- all_mortV[which(all_mortV$Year > ObsYear),]

# MinMaxScaler data pre-processing
x.min <- min(data1[[1]])
x.max <- max(data1[[1]])
x.train <- array(2*(data1[[1]]-x.min)/(x.min-x.max)-1, dim(data1[[1]]))
y.train <- - data1[[2]]
y0 <- mean(y.train)

# network architecture deep 3 network
tau1 <- 20
tau2 <- 15
tau3 <- 10
optimizer <- 'adam'

RNN.type <- "GRU"

{if (RNN.type=="LSTM"){model <- LSTM3(T0, tau0, tau1, tau2, tau3, y0, optimizer)}else{model <- GRU3(T0, tau0, tau1, tau2, tau3, y0, optimizer)}
  name.model <- paste(RNN.type,"3_", tau0, "_", tau1, "_", tau2, "_", tau3, sep="")
  file.name <- paste("./CallBack/best_model_", name.model,"_", gender, sep="")
}

# define callback
CBs <- callback_model_checkpoint(file.name, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)

# gradient descent fitting: takes roughly 200 seconds on my laptop
fit <- model %>% fit(x=x.train, y=y.train, validation_split=0.2,
                     batch_size=100, epochs=500, verbose=0, callbacks=CBs)                                        

fit_GRU = fit

# plot loss figures
plot.losses <- function(name.model, gender, val_loss, loss){
  plot(val_loss,col="cyan3", pch=20, ylim=c(0,0.1), main=list("early stopping: GRU", cex=1.5),xlab="epochs", ylab="MSE loss", cex=1.5, cex.lab=1.5)
  lines(loss,col="blue")
  legend(x="bottomleft", col=c("blue","cyan3"), lty=c(1,-1), lwd=c(1,-1), pch=c(-1,20), legend=c("in-sample loss", "out-of-sample loss"))
}   

plot.losses(name.model, gender, fit[[2]]$val_loss, fit[[2]]$loss)

# calculating in-sample loss
load_model_weights_hdf5(model, file.name)

# calculating out-of-sample loss
pred.result <- recursive.prediction(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(all_mort2$Year > ObsYear),]



#### plots of log mortality rates 2012
obs_rate_GRU = matrix(log(vali.Y$mx), nrow=100, ncol=17)
GRU_predict = matrix(log(vali$mx), nrow=100, ncol=17)
plot(obs_rate_GRU[,13], col="blue",pch=20, xlab="ages", ylab="log-morality rates", main=c("GRU vs Observed"))
lines(c(0:99),GRU_predict[,13],col='red')
legend("topleft",col=c("red","blue"), lty=c(1,-1),lwd=c(1,-1), pch=c(-1,20),legend=c("GRU", "Observed"))


#### plots of log mortality rates 2013
obs_rate_GRU = matrix(log(vali.Y$mx), nrow=100, ncol=17)
GRU_predict = matrix(log(vali$mx), nrow=100, ncol=17)
plot(obs_rate_GRU[,14], col="blue",pch=20, xlab="ages", ylab="log-morality rates", main=c("GRU vs Observed"))
lines(c(0:99),GRU_predict[,14],col='red')
legend("topleft",col=c("red","blue"), lty=c(1,-1),lwd=c(1,-1), pch=c(-1,20),legend=c("GRU", "Observed"))

