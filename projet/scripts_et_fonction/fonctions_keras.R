dessiner_frontiere_NN = function(X,y,model, xm,xM,ym,yM, cols){
  
  dev.new()
  if(length(table(y))==2){
    plot(X[which(y==0),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    points(X[which(y==1),1:2], col = cols[2])
  }
  else{
    plot(X[which(y==1),1:2], col = cols[1], xlim = c(xm,xM), ylim = c(ym,yM))
    for(j in 2:length(table(y))){
      
      points(X[which(y==j),1:2], col = cols[j])
      
    }
    
  }	
  nb = 60
  x1 = seq(xm,xM, length.out = nb)
  x2 = seq(ym, yM, length.out = nb)
  
  for(i in 1:nb){
    for(j in 1:nb){
      p= ifelse(predict(model, rbind(c(x1[i],x2[j])))>0.5,1,0)
    if(length(table(y))==2){
        if(p==0){points(x1[i],x2[j], col = cols[1], pch = 3)}
        else{points(x1[i],x2[j], col = cols[2], pch = 3)}
      }
      else{
        points(x1[i],x2[j], col = cols[p], pch = 3) 
      }
      
    }
  }	
  
}


plot_model_loss=function(history){
    dev.new()
    # Cas ou il n'y a pas d'ensemble de validation: 
    # is.vector(history$metrics$val_loss) == FALSE ou length(history$metrics$val_loss)==0
    y_max=ifelse(length(history$metrics$val_loss)==0, max(history$metrics$loss), max(max(history$metrics$loss),max(history$metrics$val_loss)))+0.05
    #y_max=max(history$metrics$val_loss)+0.05
    # Plot the model loss of the training data
    plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l",ylim=c(0, y_max))
    # Plot the model loss of the test data
    lines(history$metrics$val_loss, col="green")
    # Add legend
    legend("topright", c("train","val"), col=c("blue", "green"), lty=c(1,1))
}


plot_model_accuracy=function(history){
    dev.new()
    plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l", ylim=c(0, 1))
    lines(history$metrics$val_acc, col="green")
    legend("bottomright", c("train","val"), col=c("blue", "green"), lty=c(1,1))
}