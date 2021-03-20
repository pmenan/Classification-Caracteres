
change_label_RN = function(dataset){
  
  dataset$label[dataset$label==1]=0
  dataset$label[dataset$label==2]=1
  dataset$label[dataset$label==6]=2
  dataset$label[dataset$label==12]=3
  dataset$label[dataset$label==16]=4
  dataset$label[dataset$label==18]=5
  dataset$label[dataset$label==22]=6
  return(dataset)  
}

change_predictions_RN = function(p){
  
  p[p==6]=22
  p[p==5] = 18
  p[p==4]=16
  p[p==3] = 12
  p[p==2]=6
  p[p==1] = 2
  p[p==0] = 1
  return(p)
}
