    # X les données d'apprentissage (individus en ligne)
# y les classes. Attention : en binaire c'est 0/1, et sinon c est 1/2/3 etc
# num_layers : nombre de couches (input+output+hidden)
# layers_size : vecteur contenant les tailles des couches (input_size, hidden1, hidden2, ..., output_size)
# num_labels : nombre de classes = output_size
# Theta : liste contenant les paramètres du réseau
#	- Theta[[1]]: matrice de taille layers_size[2] x (layers_size[1]+1)
#	- Theta[[2]]: ----------------- layers_size[3] x (layers_size[2]+1) etc...
# lambda : regularization parameter


learn_nn = function(X,y, lambda, nbIter, lay, init = T, start_nn=0){
	
	if(init){
	Th_init = init_reseau(lay)
	}
	else{Th_init = start_nn}
	unroll_Th = unroll_theta(Th_init)
	xxx =optim(unroll_Th, objectif, grad,method = "BFGS",X = X, y = y, lambda=lambda, lay=lay, control = list(maxit = nbIter))
	#print(xxx)
	#if(xxx$convergence>0){print("Pb de convergence")}
	
	return(roll_theta(xxx$par, lay))
	
	
}




gradDescent = function(X, y, Theta_init, lambda,alpha, nbIter){
	
	Theta_out = Theta_init
	for(i in 1:nbIter){
		
		CostG = nnCostFunction(X, y, Theta_out, lambda);
		if(i%%1000 ==0){print(i);print(CostG[[1]])}
		for(j in 1:length(CostG[[2]])){
			
			Theta_out[[j]] = Theta_out[[j]] - alpha*CostG[[2]][[j]]
		}
		
		
		
	}
	
	return(Theta_out)
}




init_reseau = function(layers_size){
	
	Theta = list()
	if(length(layers_size)==2){
		
	Theta[[1]] = matrix(runif(layers_size[2]*(layers_size[1]+1),-0.00000001,0.00000001),nrow = layers_size[2])
		
	}
	
	else{
	
	for(i in 1:(length(layers_size)-2)){
		
		Theta[[i]] = matrix(runif(layers_size[i+1]*(layers_size[i]+1),-0.12,0.12),nrow = layers_size[i+1])
	}
	if(layers_size[length(layers_size)]==1){
		
		Theta[[length(layers_size)-1]] = matrix(runif(layers_size[length(layers_size)-1]+1,-0.12,0.12),nrow = 1)
		
	}
	else{
		i = length(layers_size)-1
		Theta[[i]] = matrix(runif(layers_size[i+1]*(layers_size[i]+1),-0.12,0.12),nrow = layers_size[i+1])
	}
	}
	
	return(Theta)
}




sigmoidGrad = function(z){
	
	return(sigmoid(z)*(1-sigmoid(z)));
	
}


objectif = function(Theta_unroll, X, y, lambda, lay){
	
	return(nnCostFunction(X,y,roll_theta(Theta_unroll,lay), lambda)[[1]])
	
	
}

grad = function(Theta_unroll, X, y, lambda, lay){
	
	return(unroll_theta(nnCostFunction(X,y,roll_theta(Theta_unroll,lay), lambda)[[2]]))
	
	
}


predict_nn = function(Theta,X){
	
	a = X
	As = list()
	Zs = list()
	Zs[[1]] = 0
	a = cbind(rep(1,nrow(a)),a)
	a = t(a)
	As[[1]] = a
	for(i in 2:(length(Theta)+1)){
		Zs[[i]] = Theta[[i-1]]%*%As[[i-1]]
		As[[i]] = sigmoid(Zs[[i]])
		As[[i]] = rbind(rep(1,ncol(As[[i]])),As[[i]])
	}
	As[[length(As)]] = As[[length(As)]][-1,]	
	
		
	p = matrix(As[[length(As)]], ncol = nrow(X))
  #return(p)
	if(nrow(p) == 1){return(ifelse(p>0.5,1,0))}
	else{
		return(apply(p,2,which.max))
	}
	
	
	
}

calcul_sorties= function(Theta,X){
  
  a = X
  As = list()
  Zs = list()
  Zs[[1]] = 0
  a = cbind(rep(1,nrow(a)),a)
  a = t(a)
  As[[1]] = a
  for(i in 2:(length(Theta)+1)){
    Zs[[i]] = Theta[[i-1]]%*%As[[i-1]]
    As[[i]] = sigmoid(Zs[[i]])
    As[[i]] = rbind(rep(1,ncol(As[[i]])),As[[i]])
  }
  As[[length(As)]] = As[[length(As)]][-1,]	
  
  
  p = matrix(As[[length(As)]], ncol = nrow(X))
  return(p)
  
  
  
  
}




nnCostFunction = function(X,y, Theta, lambda){
	
	#if(length(Theta)==1){
	#m = nrow(X)	
	#X = cbind(rep(1,m),X)
	#cost = sum(-y*log(sigmoid(Theta[[1]]%*%t(X)))-(1-y)*log(1-sigmoid(Theta[[1]]%*%t(X))))/nrow(X) + lambda *sum(Theta[[1]]^2)/(2*m) - lambda*Theta[[1]][,1]^2/(2*m);	
		
	#Theta_grad = list()	
	#Theta_grad[[1]] = apply((sigmoid(Theta[[1]]%*%t(X)) - y)*X,2,sum)/m	
	#}
	
	#else{
	
	a = X
	As = list()
	Zs = list()
	Zs[[1]] = 0
	a = cbind(rep(1,nrow(a)),a)
	a = t(a)
	As[[1]] = a
	for(i in 2:(length(Theta)+1)){
		Zs[[i]] = Theta[[i-1]]%*%As[[i-1]]
		As[[i]] = sigmoid(Zs[[i]])
		As[[i]] = rbind(rep(1,ncol(As[[i]])),As[[i]])
	}
	As[[length(As)]] = As[[length(As)]][-1,]
	#return(As)
	cost = 0;
	
	output=matrix(As[[length(As)]], ncol = nrow(X))
	
	ymat = matrix(0,nrow(output), ncol(output))
	for(i in 1:ncol(ymat)){
		ymat[,i] = as.numeric(y[i]==c(1:nrow(ymat)))
	}


	for(j in 1:ncol(output)){
		for(i in 1:nrow(output)){
			
			cost = cost - ymat[i,j]*log(output[i,j]) - (1-ymat[i,j])*log(1-output[i,j])
			
			
		}
		
	}
	cost = cost/ncol(output)
	regul = 0
	for(i in 1:length(Theta)){
		regul= regul+sum(Theta[[i]]^2) 
	}
	cost = cost+regul*(lambda/(2*ncol(output)))
	
		
	Ds = list()
	Ds[[length(As)]] = As[[length(As)]] - ymat
	
	if(length(Theta)>1){for(i in (length(Ds)-1):2){
		
		bz = t(Theta[[i]])%*%Ds[[i+1]]
		Ds[[i]] = bz[-1,]*sigmoidGrad(Zs[[i]])

		
		
	}
	}
	
	Theta_grad = list()
	
	for(i in 1:length(Theta)){
		
		Theta_grad[[i]] = matrix(0,nrow(Theta[[i]]), ncol(Theta[[i]]))
				
	}
	
	for(i in 1:length(Theta)){
		
		for(j in 1:ncol(Ds[[i+1]])){
			
			Theta_grad[[i]] = Theta_grad[[i]] + Ds[[i+1]][,j]%*%t(As[[i]][,j])
			
			
		}
		
		
		
	}	
	
	for(i in 1:length(Theta)){
	
		Theta_grad[[i]] = Theta_grad[[i]]/nrow(X)
		Theta_grad[[i]][,2:ncol(Theta_grad[[i]])] = Theta_grad[[i]][,2:ncol(Theta_grad[[i]])] + lambda*Theta[[i]][,2:ncol(Theta_grad[[i]])]/nrow(X)
	
	}
	
	#}
	
	return(list(cost, Theta_grad))
		
	
}



unroll_theta = function(Theta){
	
	out = c()
	for(i in 1:length(Theta)){
		
		out = c(out,as.vector(Theta[[i]]))
		
		
		
	}
	
	return(out)
	
	
}

roll_theta = function(vec,layers_size){
	
	out = list()
	idx = 1;
	for(i in 1:(length(layers_size)-1)){
		
		
		out[[i]] = matrix(vec[idx:(idx-1+(layers_size[i+1]*(layers_size[i]+1)))], nrow = layers_size[i+1])
		idx = idx+layers_size[i+1]*(layers_size[i]+1)
		
		
	}
	
	return(out)
	
	
}



dessiner_frontiere_NN = function(X,y,Theta, xm,xM,ym,yM, cols){
	
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
	
	x1 = seq(xm,xM, length.out = 60)
	x2 = seq(ym, yM, length.out = 60)
	
	for(i in 1:60){
			for(j in 1:60){
				p= predict_nn(Theta, rbind(c(x1[i],x2[j])))
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


dessiner_frontiere_nnet = function(nn, X,y,xm,xM,ym,yM, cols){
	
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
	
	x1 = seq(xm,xM, length.out = 60)
	x2 = seq(ym, yM, length.out = 60)
	
	for(i in 1:60){
			for(j in 1:60){
				nd = rbind(c(x1[i],x2[j]))
				colnames(nd) = colnames(X)[1:2]
				
				p= predict(nn, nd)
				if(length(table(y))==2){
				if(p<0.5){points(x1[i],x2[j], col = cols[1], pch = 3)}
				else{points(x1[i],x2[j], col = cols[2], pch = 3)}
				}
				else{
					
					points(x1[i],x2[j], col = cols[which.max(p)], pch = 3)
					
				}
				
				
				
				
				
				}
			}	
	
	
	
}







