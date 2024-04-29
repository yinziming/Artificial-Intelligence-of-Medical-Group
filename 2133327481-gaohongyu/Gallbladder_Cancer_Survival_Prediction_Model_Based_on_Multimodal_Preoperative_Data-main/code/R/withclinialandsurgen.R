surgeon<-read.xlsx('E:/R/ddm/Rcode/surgeon.xlsx')
colnames(surgeon)<-c('name','TNM_0','I','II','IIIA','IIIB','IIIx','IVA',
                      'IVB','gallstone','liver_resection_status','R0',
                      'differentiation','nerve_infiltration',
                      'microvascular_infiltration','radiation_and_chemotherapy')

#surgeon<-read.xlsx('E:/R/ddm/Rcode/surgeonwithoutonehot.xlsx')
#colnames(surgeon)<-c('name','TNM','gallstone','liver_resection_status','R0',
#                     'differentiation','nerve_infiltration',
#                     'microvascular_infiltration','radiation_and_chemotherapy')

#姓名汉字转为拼音并设置为行名
mypy <- pydic(method = 'toneless',dic='pinyin2') # 载入默认字典
a<-sapply(list(surgeon$name),function(x){py(x,sep = '',dic = mypy)})
surgeon$pname<-a[,1]
surgeon[!duplicated(surgeon$pname),]->surgeon
rownames(surgeon)<-surgeon$pname

# -----------------------------------------------------
xinhua_clinical<-merge_survival_data[, c(1:23)]
renji_clinical<-final_merge[,c(1:22)]
colnames(xinhua_clinical)
colnames(renji_clinical)
a<-xinhua_clinical[,c(1, 2, 9:23)]
a<-na.omit(a)
colnames(a)
for(i in colnames(a[,c(3:17)]))
{
  a[,i]<-as.numeric(a[,i])
}
b<-renji_clinical[,c(1,2,8,9,10:22)]
colnames(b)<-colnames(a)
final_clinical<-rbind(a,b)
ff<-merge(final_clinical,risk_score_table_multi_cox2[,c(1,2,3,12)],by.x='pname',by.y='pname',all = F)

library(mice)
final_c_s=mice(ff,seed = 1234)
final_c_s=complete(final_c_s,action = 2)

final_c_s <- merge(final_c_s, surgeon,by.x='pname',by.y='pname',all = F)

#生存分析
library(survival)
library(dplyr)
library(plyr)
library(ggplot2)
library(survminer)
S.OS=with(final_c_s,Surv(time,status))
UniCox<-function(x){
  FML<-as.formula(paste0('S.OS~',x))
  # Cox<-coxph(FML,data=data)
  #b<-Surv(OS.time,OS)
  #FML<-as.formula(paste0('b~',x))
  cox<-coxph(formula = FML,data=final_c_s,x=T,y=T)
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
#单因素分析
VarNames2<-colnames(final_c_s)[c(3:17,22:36)]
#VarNames2<-colnames(final_c_s)[c(3:17,22:29)]
UniVar2<-lapply(VarNames2,UniCox)
UniVar2<-ldply(UniVar2,data.frame)
clinical_Var2<-subset(UniVar2,P.Value<0.05)   #删选pvalue<0.05的临床特征
clinical_Var2

#临床加影像
library(data.table)
setnames(final_c_s,"total_risk_score","deep_feature")
ddist=datadist(final_c_s)
options(datadist="ddist")
f.final<-cph(Surv(time,status)~CEA+deep_feature+CA199+TNM_0,x=T,y=T,surv = T,data = final_c_s)
survival<-Survival(f.final)
survival1<-function(x)survival(1095,x)
survival2<-function(x)survival(1825,x)

nom<-nomogram(f.final,fun=list(survival1,survival2),
              fun.at = c(0.05,seq(0.1,0.9,by=0.05),0.95),
              funlabel = c('3year','5year'))
plot(nom)

cindex = rcorr.cens(predict(f.final), S.OS) 
1 - cindex[1]
boot.val = validate(f.final, method="boot", B = 1000)
boot.val
boot.c = 0.5 + abs(boot.val["Dxy","index.corrected"])/2 
boot.c
final_c_s
final_c_s$marker=1.352747e-03*final_c_s$CEA+8.309631e-01*final_c_s$deep_feature+4.952385e-05*final_c_s$CA199+2.131712*final_c_s$TNM_0+1.906138e-01*final_c_s$Fib

#regplot(f.final,observation = final_c_s[2,], clickable = T,points = T,failtime = c(36,60),plots = c("density", "boxes"))
regplot(f.final,observation = FALSE, clickable = T,points = T,failtime = c(36,60),plots = c("density", "boxes"))

#绘制ROC

roc1=survivalROC(Stime = final_c_s$time,status = final_c_s$status,marker = final_c_s$marker,predict.time = 1*365,method = 'KM')
roc3=survivalROC(Stime = final_c_s$time,status = final_c_s$status,marker = final_c_s$marker,predict.time = 3*365,method = 'KM')
roc5=survivalROC(Stime = final_c_s$time,status = final_c_s$status,marker = final_c_s$marker,predict.time = 5*365,method = 'KM')
plot(roc1$FP,roc1$TP,type = 'l',xlim = c(0,1),ylim=c(0,1),col = 'red',
     xlab = 'False positive rate',ylab = 'True positive rate',
     main = 'ROC curve',
     lwd = 2,cex.main = 1.3,cex.lab = 1.2,cex.axis = 1.2,font = 1.2)

lines(roc3$FP,roc3$TP,lty=1,lwd = 2,col = '#2E9FDF')
lines(roc5$FP,roc5$TP,lty=1,lwd = 2,col = '#E7B800')
abline(0,1,lty=2)
legend("bottomright", legend=c(paste('1 years','AUC = ',round(roc1$AUC,3)),paste('3 years','AUC = ',round(roc3$AUC,3)),paste('5 years','AUC = ',round(roc5$AUC,3))), col=c("red","#2E9FDF",'#E7B800'),lty=1,lwd=2)

roc1_df = as.data.frame(roc1)
roc3_df = as.data.frame(roc3)
roc5_df = as.data.frame(roc5)

wb <- createWorkbook()
addWorksheet(wb, sheetName = "Sheet 1")
addWorksheet(wb, sheetName = "Sheet 3")
addWorksheet(wb, sheetName = "Sheet 5")

writeData(wb, sheet = 1, roc1_df)
writeData(wb, sheet = 2, roc3_df)
writeData(wb, sheet = 3, roc5_df)

saveWorkbook(wb, 'E:/R/ddm/Rcode/RocDataTrain.xlsx', overwrite = TRUE)

riskscore2<-function(survival_cancer_df,candidate_genes_for_cox){
  risk_score_table<-survival_cancer_df[,candidate_genes_for_cox]
  for(each_sig_gene in colnames(risk_score_table)){
    risk_score_table[,each_sig_gene]<-risk_score_table[,each_sig_gene]*(as.data.frame(f.final$coefficients)[each_sig_gene,1])
  }
  risk_score_table<-cbind(risk_score_table,'total_risk_score'=rowSums(risk_score_table))%>%
    cbind(survival_cancer_df[,c('pname','time','status')])
  risk_score_table<-risk_score_table[,c('pname','time','status',candidate_genes_for_cox,'total_risk_score')]
}
candidate_gene_for_cox2<-c(names(f.final$coefficients))
risk_score_table_multi_cox2<-riskscore2(final_c_s,candidate_gene_for_cox2)


info_hugo<-risk_score_table_multi_cox2%>%
  mutate(
    risk_Status=ifelse(total_risk_score>median(total_risk_score,na.rm = TRUE),'High',
                       ifelse(is.na(total_risk_score),'NA','Low'))
  )

fit_final<- survfit(Surv(time, status) ~ risk_Status, data = info_hugo)

med_cut_off<-median(info_hugo$total_risk_score,na.rm = TRUE)

ggsurvplot(fit_final,
           data = info_hugo, conf.int = T,pval = TRUE, fun = "pct",
           xlab = "Time (in days)",risk.table = T,
           palette = c('#E7B800','#2E9FDF'), ggtheme = theme_light(), ncensor.plot = TRUE,
           xlim = c(0,1500)
)

wb <- createWorkbook()
addWorksheet(wb, sheetName = "Sheet 1")
writeData(wb, sheet = 1, UniVar2)
saveWorkbook(wb, 'E:/R/ddm/Rcode/UniVar2.xlsx', overwrite = TRUE)