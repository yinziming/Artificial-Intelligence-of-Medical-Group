setwd('E:/R')
library(openxlsx)
library(pinyin)
library(dplyr)
library(survival)
library(dplyr)
library(plyr)
library(ggplot2)
library(survminer)
library(glmnet)
library(rms)
library(regplot)
library(survivalROC)
library(ggDCA)
library(ggprism)

clinical<-read.xlsx('E:/R/ddm/Rcode/clinicalRenji.xlsx')
colnames(clinical)<-c('name','sex','age','PT','INR','APTT','Fib','TT',
                      'white_cell','oxyphorase','soterocyte','AFP',
                      'CEA','CA199','CA125','CA724')

#姓名汉字转为拼音并设置为行名

mypy <- pydic(method = 'toneless',dic='pinyin2') # 载入默认字典
a<-sapply(list(clinical$name),function(x){py(x,sep = '',dic = mypy)})
clinical$pname<-a[,1]
clinical[!duplicated(clinical$pname),]->clinical
rownames(clinical)<-clinical$pname

#数据处理

clinical<-clinical[,-1] #去掉第一列name
clinical<-clinical[,-16]#去掉最后一列pname
clinical[which(clinical$sex=='男'),'sex']<-1 #性别二值化
clinical[which(clinical$sex=='女'),'sex']<-0
for(i in colnames(clinical))
{
  clinical[,i]<-as.numeric(clinical[,i])
}

#插值，删失
library(mice)
imp=mice(clinical,seed = 1234)
data=complete(imp,action = 2)
data$name<-rownames(data)

#读取随访
survival2<-read.xlsx('E:/R/ddm/Rcode/followupRenji.xlsx')
colnames(survival2)<-c('name','sex','age','id','status','time')
a<-sapply(list(survival2$name),function(x){py(x,sep = '',dic = mypy)})
survival2$pname<-a[,1]
survival2[!duplicated(survival2$pname),]->survival2
rownames(survival2)<-survival2$pname
merge_os<-merge(survival2,data,by.x = 'pname',by.y='name',all = F)
#test<-merge(survival2,data,by.x = 'pname',by.y='name',all.x = TRUE)
merge_os[which(merge_os$status=='die'),'status']<-1
merge_os[which(merge_os$status=='live'),'status']<-0
merge_os$status<-as.numeric(merge_os$status)
#读取影像学的特征
renji_data<-read.csv('E:/R/ddm/Rcode/renji6.csv')
gsub('^[0-9]','',renji_data$X)->renji_data$X
gsub('^[0-9]','',renji_data$X)->renji_data$X
gsub('^[0-9]','',renji_data$X)->renji_data$X
substring(renji_data$X,1,nchar(renji_data$X)-4)->renji_data$X
final_merge<-merge(merge_os,renji_data,by.x = 'pname',by.y='X',all = F)
final_renji<-final_merge[,c(1,6,7,23:2070)]
final_data<-transform(final_renji,time = time/30)

S.OS=with(final_data,Surv(time,status))

UniCox<-function(x){
  FML<-as.formula(paste0('S.OS~',x))
  # Cox<-coxph(FML,data=data)
  #b<-Surv(OS.time,OS)
  #FML<-as.formula(paste0('b~',x))
  
  cox<-coxph(formula = FML,data=final_data,x=T,y=T)
  Gsum<-summary(cox)
  Hr<-round(Gsum$coefficients[,2],6)
  pvalue<-round(Gsum$coefficients[,5],10)
  lower<-round(Gsum$conf.int[,3],4)
  upper<-round(Gsum$conf.int[,4],4)
  CI<-paste0(round(Gsum$conf.int[,3:4],4),collapse = '-')
  coefficients<-cox$coefficients
  Unicox<-data.frame('characteristics'=x,'Hazard Ratio'=Hr,'lower'=lower,'upper'=upper,'CI95'=CI,'P Value'=pvalue,'coefficients'=coefficients)
  return(Unicox)
}

VarNames<-colnames(final_data)[c(4:2051)]
UniVar2<-lapply(VarNames,UniCox)
UniVar2<-ldply(UniVar2,data.frame)
Var2<-subset(UniVar2,P.Value<0.05)

#多因素分析

x<-as.matrix(final_data[,Var2$characteristics])
y<-as.matrix(final_data[,c('time','status')])
colnames(y)<-c('time','status')

###lasso
lasso_fit<-cv.glmnet(x,y,family='cox',type.measure = 'deviance', alpha = 1, nfolds = 10)
plot(lasso_fit)
fit<-glmnet(x,y,family = 'cox')
plot(fit)
cofficients<-coef(lasso_fit,s=lasso_fit$lambda.min)
Active.Index<-which(as.numeric(cofficients)!=0)
active.cofficients<-as.numeric(cofficients)[Active.Index]
sig_gene_multi_cox<-rownames(cofficients)[Active.Index]

sig_gene_multi_cox

#随机森林
library(randomForestSRC)
library(rms)
library(lattice)
rfdata<-final_data[,c('time','status',sig_gene_multi_cox)]
rf<-rfsrc(Surv(time,status)~.,data=rfdata,nsplit = 10,ntree = 500,importance = T,nodesize = 2)
test=rf[['importance']]
test=subset(test,test>0.005)
test
yvar <- rf$yvar
trellis.device(device="windows", height = 20, width = 20, color=TRUE)
plot(rf)
# formula_for_multivariate<-as.formula('Surv(time,status) ~ X367 + X658 + X828 + X659 +X661')


formula_for_multivariate<-as.formula(paste0('Surv(time,status)~',paste(sig_gene_multi_cox,sep = '',collapse = '+')))
survival_data<-cbind(x,y)
survival_data<-as.data.frame(survival_data)
ddist=datadist(survival_data)
options(datadist="ddist")
f.step<-cph(formula_for_multivariate,data = survival_data,x=T,y=T)


# f.m1<-cph(formula_for_multivariate,data = survival_data,x=T,y=T)
# f.step=step(f.m1,direction='backward')


f.step
#formula_for_multivariate<-as.formula('Surv(time,status) ~ X367 + X658 + X659 +X840')
cindex = rcorr.cens(predict(f.step), S.OS) 
1 - cindex[1] #C-indexΪ0.6792759
CstatisticCI <- function(x) {            #C-index 95%CI 
  se <- x["S.D."]/2    #others x["S.D."]/sqrt(x["n"])
  Low95 <- x["C Index"] - 1.96*se 
  Upper95 <- x["C Index"] + 1.96*se 
  cbind(x["C Index"], Low95, Upper95) 
} 
1 - CstatisticCI(cindex) 
boot.val = validate(f.step, method="boot", B = 1000)
boot.val
boot.c = 0.5 + abs(boot.val["Dxy","index.corrected"])/2 
boot.c


##
####
riskscore2<-function(survival_cancer_df,candidate_genes_for_cox){
  risk_score_table<-survival_cancer_df[,candidate_genes_for_cox]
  for(each_sig_gene in colnames(risk_score_table)){
    risk_score_table[,each_sig_gene]<-risk_score_table[,each_sig_gene]*(as.data.frame(f.step$coefficients)[each_sig_gene,1])
  }
  risk_score_table<-cbind(risk_score_table,'total_risk_score'=rowSums(risk_score_table))%>%
    cbind(survival_cancer_df[,c('pname','time','status')])
  risk_score_table<-risk_score_table[,c('pname','time','status',candidate_genes_for_cox,'total_risk_score')]
}
candidate_gene_for_cox2<-c(names(f.step$coefficients))
risk_score_table_multi_cox2<-riskscore2(final_data,candidate_gene_for_cox2)


info_hugo5<-risk_score_table_multi_cox2%>%
  mutate(
    risk_Status=ifelse(total_risk_score>median(total_risk_score,na.rm = TRUE),'High',
                       ifelse(is.na(total_risk_score),'NA','Low'))
  )

fit4<- survfit(Surv(time, status) ~ risk_Status, data = info_hugo5)

med_cut_off<-median(info_hugo5$total_risk_score,na.rm = TRUE)

# ggsurvplot(fit4,
#            data = info_hugo5, pval = TRUE, fun = "pct",
#            xlab = "Time (in days)-train",risk.table = T
# )
ggsurvplot(fit4,
           data = info_hugo5, conf.int = T,pval = TRUE, fun = "pct",
           xlab = "Time (in month)-train",risk.table = T,
           palette = c('#E7B800','#2E9FDF'), ggtheme = theme_light(), ncensor.plot = TRUE,
           xlim = c(0,60)
)

#绘制列线图

f.step<-cph(formula_for_multivariate,data = survival_data,x=T,y=T)
regplot(f.step,observation = info_hugo5[1,], clickable = T,points = T,failtime = c(36,60),plots = c("density", "boxes"))

#绘制ROC

roc1=survivalROC(Stime = info_hugo5$time,status = info_hugo5$status,marker = info_hugo5$total_risk_score,predict.time = 1*12,method = 'KM')
roc3 = survivalROC(Stime = info_hugo5$time,status = info_hugo5$status,marker = info_hugo5$total_risk_score,predict.time = 3*12,method = 'KM')
roc5 = survivalROC(Stime = info_hugo5$time,status = info_hugo5$status,marker = info_hugo5$total_risk_score,predict.time = 5*12,method = 'KM')
plot(roc1$FP,roc1$TP,type = 'l',xlim = c(0,1),ylim=c(0,1),col = 'red',
     xlab = 'False positive rate',ylab = 'True positive rate',
     main = 'ROC curve',
     lwd = 2,cex.main = 1.3,cex.lab = 1.2,cex.axis = 1.2,font = 1.2)

lines(roc3$FP,roc3$TP,lty=1,lwd = 2,col = '#2E9FDF')
lines(roc5$FP,roc5$TP,lty=1,lwd = 2,col = '#E7B800')
abline(0,1,lty=2)
legend("bottomright", legend=c(paste('1 years','AUC = ',round(roc1$AUC,3)),paste('3 years','AUC = ',round(roc3$AUC,3)),paste('5 years','AUC = ',round(roc5$AUC,3))), col=c("red","#2E9FDF",'#E7B800'),lty=1,lwd=2)




