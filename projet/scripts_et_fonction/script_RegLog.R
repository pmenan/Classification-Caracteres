
change_label_RegLog = function(dataset){
  
  dataset$label[dataset$label==6]=3
  dataset$label[dataset$label==12]=4
  dataset$label[dataset$label==16]=5
  dataset$label[dataset$label==18]=6
  dataset$label[dataset$label==22]=7
  return(dataset)  
}

change_predictions_RegLog = function(p){
  
  p[p==7]=22
  p[p==6] = 18
  p[p==5]=16
  p[p==4] = 12
  p[p==3]=6
  
  return(p)
}
