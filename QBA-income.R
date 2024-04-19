
# Probabilistic bias analysis for exposure misclassification of household income by neighbourhood in a cohort of colorectal cancer patients 
# Laura E Davis, PhD, Hailey R Banack, PhD, Renzo Calderon-Anyosa MD, PhD, Erin C Strumpf, PhD, Alyson L Mahar PhD

# R code for quantitative bias analysis, using simulated data 

# misclassified measured exposure = neighbourhood income quintile 
# true exposure = household income quintile
# outcome = 5-year mortality (dichotomous)
# predictors of misclassification = age, sex, rural residence 
# confounders = age, sex, rural residence 



# SET UP ------------------------------------------------------------------


## SUMMARY OF MODEL OUTPUT ##
#- full cohort, 
#- beta distribution using confidence intervals from the model 
#- ppvs by outcome  
#- 10,000 iterations 

rm(list=ls())
options(scipen = 999)

library(multcomp)
library(episensr)
library(tidyverse)
library(nnet)
library(marginaleffects)
library(dplyr)
library(effects)
library(trapezoid)
library(geepack)
library(broom)
library(ggplot2)
library(truncnorm)

# Simulate data -----------------------------------------------------------

N <- 21595

# household income 
ind_inc <- abs(rnorm(N, 53850, 26000))
ind <- as.factor(cut_number(ind_inc, n = 5, labels=1:5))

# neighbourhood income 
neighb_inc <- abs(rnorm(N, 54120, 10000))
neighb <- as.factor(cut_number(neighb_inc, n = 5, labels=1:5))

# province/territories
prov <- sample(c('BC', 'PR', 'ON', 'QC', 'AT'), N, replace=T, prob=c(0.14, 0.18, 0.41, 0.16, 0.11))

# age and sex
age <- rtruncnorm(n=N, a=35, b=105, mean=66, sd=12)
sex <- rbinom(N, 1, prob=0.55)
rural <- rbinom(N, 1, prob=0.24)
person_ID <- 1:21595

# 5-year survival 
dead_5yr <- rbinom(N, 1, 0.42)

# combine 
D <- data.frame(ind, neighb, prov, age, sex, rural, dead_5yr, person_ID)

# dummy  neighb income 
D$neighb_1 <- ifelse(D$neighb==1, 1, 0)
D$neighb_2 <- ifelse(D$neighb==2, 1, 0)
D$neighb_3 <- ifelse(D$neighb==3, 1, 0)
D$neighb_4 <- ifelse(D$neighb==4, 1, 0)

# dummy  ind income 
D$ind_1 <- ifelse(D$ind==1, 1, 0)
D$ind_2 <- ifelse(D$ind==2, 1, 0)
D$ind_3 <- ifelse(D$ind==3, 1, 0)
D$ind_4 <- ifelse(D$ind==4, 1, 0)




# get confint -- for later
confint.geeglm <- function(object, parm, level = 0.95, ...) {
  cc <- coef(summary(object))
  mult <- qnorm((1+level)/2)
  citab <- with(as.data.frame(cc),
                cbind(lwr=Estimate-mult*Std.err,
                      upr=Estimate+mult*Std.err))
  rownames(citab) <- rownames(cc)
  citab[parm,]
}


# 1. Bias parameters ------------------------------------------------------

# -------------------------------------------------------------------------#
# Step 2. GET PREDICTIVE VALUES FROM MODEL 
# Neighbourhood income as measured and biased, individual income as true 
# to get predictive values we reverse the model (ind=outcome)
# - alive 
# - dead 
# - alive/dead by province 
# for each: 
# 1. crude 
# 2. adjust for rural + age + sex 
# -------------------------------------------------------------------------#

# make 5x5 table 
tab <- table(D$neighb,D$ind) # first is row variables and second is column, i.e. individual is at the top

# get crude predictive values from the table
tab.pv <- tab/rowSums(tab)
tab.pv

# 5x5 table and predictive values by status
D_alive <- D %>%  
  filter(dead_5yr==0)
D_dead <- D %>% 
  filter(dead_5yr==1)
tab.D0 <- table(D_alive$neighb, D_alive$ind)
tab.D1 <- table(D_dead$neighb, D_dead$ind)
tab.D0.pv <- tab.D0/rowSums(tab.D0)
tab.D1.pv <- tab.D1/rowSums(tab.D1)
tab.D0
tab.D1

# -------------------------------------------------------------------------#
##### CRUDE MODELS FOR PREDICTIVE VALUES ##### 
# these match the 5x5 table 
# -------------------------------------------------------------------------#

# create list with all datasets to loop through 
# 65 = territories, 70 = atlantic provinces, 80 = prairie provinces
# 24 = quebec, 35 = ontario, 59 = BC
dlist <- list()
dlist[[1]] <- D
dlist[[2]] <- D_alive
dlist[[3]] <- D_dead
dlist[[4]] <- D %>%  
  filter(prov=='QC')
dlist[[5]] <- D_alive %>%  
  filter(prov=='QC')
dlist[[6]] <- D_dead %>%  
  filter(prov=='QC')
dlist[[7]] <- D %>%  
  filter(prov=='ON')
dlist[[8]] <- D_alive %>%  
  filter(prov=='ON')
dlist[[9]] <- D_dead %>%  
  filter(prov=='ON')
dlist[[10]] <- D %>%  
  filter(prov=='BC')
dlist[[11]] <- D_alive %>%  
  filter(prov=='BC')
dlist[[12]] <- D_dead %>%  
  filter(prov=='BC')
dlist[[13]] <- D %>%  
  filter(prov=='PR')
dlist[[14]] <- D_alive %>%  
  filter(prov=='PR')
dlist[[15]] <- D_dead %>%  
  filter(prov=='PR')
dlist[[16]] <- D %>%  
  filter(prov=='AT')
dlist[[17]] <- D_alive %>%  
  filter(prov=='AT')
dlist[[18]] <- D_dead %>%  
  filter(prov=='AT')

names(dlist) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                  "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")


# 1.1 Crude ---------------------------------------------------------------

# Save model output as list for each dataset 
mlist <- lapply(dlist, function(x) multinom(ind~neighb, data=x))

# use model output to create and save pvs, lower and upper cis
crudefit <- lapply(mlist, function(x) Effect("neighb", x))
crude.pv <- sapply(1:18, function(x) as.vector(t(crudefit[[x]]$prob)))
rownames(crude.pv) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                        "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                        "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                        "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                        "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")
colnames(crude.pv) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                        "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")


crude.lower <- sapply(1:18, function(x) as.vector(t(crudefit[[x]]$lower.prob)))
colnames(crude.lower) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                           "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")
rownames(crude.lower) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                           "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                           "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                           "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                           "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")

crude.upper <- sapply(1:18, function(x) as.vector(t(crudefit[[x]]$upper.prob)))
colnames(crude.upper) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                           "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")
rownames(crude.upper) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                           "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                           "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                           "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                           "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")

# save 

# -------------------------------------------------------------------------#
# MODELS FOR PREDICTIVE VALUES -- ADJUSTED FOR RURAL RESIDENCE + AGE AND SEX # 
# create model with rural residence + age and sex 
# -------------------------------------------------------------------------#

# 1.2 Adjusted ------------------------------------------------------------


# Save model output as list for each dataset 
mlist_adj <- lapply(dlist, function(x) multinom(ind~neighb+rural+sex+age, data=x))

# use model output to create and save pvs, lower and upper cis
adjfit <- lapply(mlist_adj, function(x) Effect("neighb", x))

adj.pv <- sapply(1:18, function(x) as.vector(t(adjfit[[x]]$prob)))
colnames(adj.pv) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                      "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")
rownames(adj.pv) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                      "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                      "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                      "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                      "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")

adj.lower <- sapply(1:18, function(x) as.vector(t(adjfit[[x]]$lower.prob)))
colnames(adj.lower) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                         "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")
rownames(adj.lower) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                         "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                         "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                         "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                         "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")

adj.upper <- sapply(1:18, function(x) as.vector(t(adjfit[[x]]$upper.prob)))
colnames(adj.upper) <- c("D", "D_alive", "D_dead", "D_QC", "D_QC_alive", "D_QC_dead", "D_ON", "D_ON_alive", "D_ON_dead", 
                         "D_BC", "D_BC_alive", "D_BC_dead", "D_PR",  "D_PR_alive", "D_PR_dead","D_AT", "D_AT_alive", "D_AT_dead")
rownames(adj.upper) <- c("IQ1.NQ1", "IQ2.NQ1", "IQ3.NQ1", "IQ4.NQ1", "IQ5.NQ1", 
                         "IQ1.NQ2", "IQ2.NQ2", "IQ3.NQ2", "IQ4.NQ2", "IQ5.NQ2", 
                         "IQ1.NQ3", "IQ2.NQ3", "IQ3.NQ3", "IQ4.NQ3", "IQ5.NQ3", 
                         "IQ1.NQ4", "IQ2.NQ4", "IQ3.NQ4", "IQ4.NQ4", "IQ5.NQ4", 
                         "IQ1.NQ5", "IQ2.NQ5", "IQ3.NQ5", "IQ4.NQ5", "IQ5.NQ5")


# save

# make into a lists 
pvlist <- list(crude.pv, adj.pv)
names(pvlist) <- c("crude", "adjusted")
lowerlist <- list(crude.lower, adj.lower)
names(lowerlist) <- c("crude", "adjusted")
upperlist <- list(crude.upper, adj.upper)
names(upperlist) <- c("crude", "adjusted")



# 5x5 tables --------------------------------------------------------------

tables5x5 <- lapply(1:18, function(x) table(dlist[[x]]$neighb, dlist[[x]]$ind))



# -------------------------------------------------------------------------#
# Step 3. GET VALUES TO SAMPLE PREDICTIVE VALUES 
# Beta distribution 
# use confidence intervals from adjusted PV as upper and lower values  
# -------------------------------------------------------------------------#


# Beta distribution -------------------------------------------------------

# beta distribution, fuction to get parameters 
sd <- function(l,u){
  (u - l)/(2*1.96)
}
alpha <- function(x,sd){
  x*((x*(1-x)/sd^2)-1)
}
beta <- function(x, sd){
  (1-x)*((x*(1-x)/sd^2)-1)
}

# GET SD FOR ALL #
sdlist <- list()

# crude 
sdlist[[1]] <- sd(crude.lower, crude.upper)

# adjusted
sdlist[[2]] <- sd(adj.lower, adj.upper)

# name and save
names(sdlist) <- c("crude", "adjusted")


# GET ALPHA FOR ALL # 
alphalist <- list()

# crude 
alphalist[[1]] <- alpha(crude.pv, sdlist[[1]])
# adjusted
alphalist[[2]] <- alpha(adj.pv, sdlist[[2]])

# name and save
names(alphalist) <- c("crude", "adjusted")


# GET BETA FOR ALL #
betalist <- list()

# crude 
betalist[[1]] <- beta(crude.pv, sdlist[[1]])
# adjusted
betalist[[2]] <- beta(adj.pv, sdlist[[2]])

# name and save
names(betalist) <- c("crude", "adjusted")





# 1.QBA - Canada -----------------------------------------------------

# set up data 
I=10
o=rep(0,I)
N <- 21595
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}
pv_rbeta.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_rbeta.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_alive'], betalist[[2]][x,'D_alive']))
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_dead'], betalist[[2]][x,'D_dead']))
    
    # reassign neighbourhood income based on distribution above 
    D_new <- dlist[['D']] %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_rbeta.D0[i, 1:25] <- rb.D0
    pv_rbeta.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}

# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D02 <- data.frame(sapply(1:25, function(x) quantile(pv_rbeta.D0[,x],c(.025,.5,.975))))
sum.ppv.D02[4,] <- adj.pv[,'D_alive']
sum.ppv.D02[5,] <- adj.lower[,'D_alive']
sum.ppv.D02[6,] <- adj.upper[,'D_alive']
row.names(sum.ppv.D02)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D02) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D12 <- data.frame(sapply(1:25, function(x) quantile(pv_rbeta.D1[,x],c(.025,.5,.975))))
sum.ppv.D12[4,] <- adj.pv[,'D_dead']
sum.ppv.D12[5,] <- adj.lower[,'D_dead']
sum.ppv.D12[6,] <- adj.upper[,'D_dead']
row.names(sum.ppv.D12)[4:6] <- c("value", "lower", "upper") 
colnames(sum.ppv.D12) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

# summarise output
# make sure there are no weird values 
output2 <- output %>% filter(se_q1!=0 & se_q1<100 |
                               se_q2!=0 & se_q2<100 |
                               se_q3!=0 & se_q3<100 |
                               se_q4!=0 & se_q4<100 )
# calculate quintiles with error
q1_e<-quantile(output2$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output2$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output2$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output2$rr_q4_tot,c(.025,.5,.975))
# calculate quintiles without random error
q1<-quantile(output2$rr_q1,c(.025,.5,.975))
q2<-quantile(output2$rr_q2,c(.025,.5,.975))
q3<-quantile(output2$rr_q3,c(.025,.5,.975))
q4<-quantile(output2$rr_q4,c(.025,.5,.975))

results1 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results2) <- c("2.5_diffbeta_full","50_diffbeta_full", "97.5_diffbeta_full")


## COMPARE TO ALL OF CANADA ##
# neighbourhood model 
obs=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
           data    = D,
           family  = poisson(link = "log"),
           id      = person_ID,
           corstr  = "exchangeable")

# save output to compare
results1$RR_obs <- exp(coef(obs))[2:5]
results1$lwr_obs <- exp(confint.geeglm(obs))[2:5,1]
results1$upr_obs <- exp(confint.geeglm(obs))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
            data    = D,
            family  = poisson(link = "log"),
            id      = person_ID,
            corstr  = "exchangeable")

# save output to compare
results1$RR_true <- exp(coef(true))[2:5]
results1$lwr_true <- exp(confint.geeglm(true))[2:5,1]
results1$upr_true <- exp(confint.geeglm(true))[2:5,2]

results1




# 4.QBA - Ontario ----------------------------------------------------

# set up data 
I=10
o=rep(0,I)
N <- 8955
D_ON <- D %>% filter(prov=='ON')
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}

pv_triang.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_triang.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1,  alphalist[[2]][x,'D_ON_alive'], betalist[[2]][x,'D_ON_alive'])) 
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_ON_dead'], betalist[[2]][x,'D_ON_dead'])) 
    
    # reassign neighbourhood income based on distribution above 
    D_new <- D_ON %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_triang.D0[i, 1:25] <- rb.D0
    pv_triang.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}

# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D04 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D0[,x],c(.025,.5,.975))))
sum.ppv.D04[4,] <- adj.pv[,'D_ON_alive']
sum.ppv.D04[5,] <- adj.lower[,'D_ON_alive']
sum.ppv.D04[6,] <- adj.upper[,'D_ON_alive']
row.names(sum.ppv.D04)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D04) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D14 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D1[,x],c(.025,.5,.975))))
sum.ppv.D14[4,] <- adj.pv[,'D_ON_dead']
sum.ppv.D14[5,] <- adj.lower[,'D_ON_dead']
sum.ppv.D14[6,] <- adj.upper[,'D_ON_dead']
row.names(sum.ppv.D14)[4:6] <- c("pv", "lower", "upper")  
colnames(sum.ppv.D14) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")


# summarise output
# make sure there are no weird values 
output4 <- output %>% filter(se_q1!=0 & se_q1<100 |
                               se_q2!=0 & se_q2<100 |
                               se_q3!=0 & se_q3<100 |
                               se_q4!=0 & se_q4<100 )
# calculate quintiles with error
q1_e<-quantile(output4$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output4$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output4$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output4$rr_q4_tot,c(.025,.5,.975))

# calculate quintiles without error
q1<-quantile(output4$rr_q1,c(.025,.5,.975))
q2<-quantile(output4$rr_q2,c(.025,.5,.975))
q3<-quantile(output4$rr_q3,c(.025,.5,.975))
q4<-quantile(output4$rr_q4,c(.025,.5,.975))

results4 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results4) <- c("2.5_adjRR","50_adjRR", "97.5_adjRR")
results4

## COMPARE ONTARIO ##
# neighbourhood model for ON 
obs_ON=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
              data    = D_ON,
              family  = poisson(link = "log"),
              id      = person_ID,
              corstr  = "exchangeable")

# save output to compare
results4$RR_obs_ON <- exp(coef(obs_ON))[2:5]
results4$lwr_obs_ON <- exp(confint.geeglm(obs_ON))[2:5,1]
results4$upr_obs_ON <- exp(confint.geeglm(obs_ON))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true_ON=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
               data    = D_ON,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")

# save output to compare
results4$RR_true_ON <- exp(coef(true_ON))[2:5]
results4$lwr_true_ON <- exp(confint.geeglm(true_ON))[2:5,1]
results4$upr_true_ON <- exp(confint.geeglm(true_ON))[2:5,2]

results4






# 6.QBA Quebec -------------------------------------------------------

# set up data 
I=10
o=rep(0,I)
N <- 3461
D_QC <- D %>% filter(prov=='QC')
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}

pv_triang.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_triang.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_QC_alive'], betalist[[2]][x,'D_QC_alive'])) 
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_QC_dead'], betalist[[2]][x,'D_QC_dead'])) 
    
    # reassign neighbourhood income based on distribution above 
    D_new <- D_QC %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_triang.D0[i, 1:25] <- rb.D0
    pv_triang.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}

# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D06 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D0[,x],c(.025,.5,.975))))
sum.ppv.D06[4,] <- adj.pv[,'D_QC_alive']
sum.ppv.D06[5,] <- adj.lower[,'D_QC_alive']
sum.ppv.D06[6,] <- adj.upper[,'D_QC_alive']
row.names(sum.ppv.D06)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D06) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D16 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D1[,x],c(.025,.5,.975))))
sum.ppv.D16[4,] <- adj.pv[,'D_QC_dead']
sum.ppv.D16[5,] <- adj.lower[,'D_QC_dead']
sum.ppv.D16[6,] <- adj.upper[,'D_QC_dead']
row.names(sum.ppv.D16)[4:6] <- c("PV", "lower", "upper") 
colnames(sum.ppv.D16) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")


# summarise output
# make sure there are no weird values 
output6 <- output %>% filter(se_q1!=0 & se_q1<100 |
                               se_q2!=0 & se_q2<100 |
                               se_q3!=0 & se_q3<100 |
                               se_q4!=0 & se_q4<100 )
# calculate quintiles 
q1_e<-quantile(output6$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output6$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output6$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output6$rr_q4_tot,c(.025,.5,.975))
# calculate quintiles without error
q1<-quantile(output6$rr_q1,c(.025,.5,.975))
q2<-quantile(output6$rr_q2,c(.025,.5,.975))
q3<-quantile(output6$rr_q3,c(.025,.5,.975))
q4<-quantile(output6$rr_q4,c(.025,.5,.975))

results6 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results6) <- c("2.5_adj","50_adj", "97.5_adj")


## COMPARE QUEBEC ##
# neighbourhood model 
obs_qc=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
              data    = D_QC,
              family  = poisson(link = "log"),
              id      = person_ID,
              corstr  = "exchangeable")

# save output to compare
results6$RR_obs_QC <- exp(coef(obs_qc))[2:5]
results6$lwr_obs_QC <- exp(confint.geeglm(obs_qc))[2:5,1]
results6$upr_obs_QC <- exp(confint.geeglm(obs_qc))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true_QC=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
               data    = D_QC,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")

# save output to compare
results6$RR_true_QC <- exp(coef(true_QC))[2:5]
results6$lwr_true_QC <- exp(confint.geeglm(true_QC))[2:5,1]
results6$upr_true_QC <- exp(confint.geeglm(true_QC))[2:5,2]
results6




# 8.QBA BC -----------------------------------------------------------

# set up data 
I=10
o=rep(0,I)
N <- 3003
D_BC <- D %>% filter(prov=='BC')
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}

pv_triang.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_triang.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_BC_alive'], betalist[[2]][x,'D_BC_alive'])) 
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_BC_dead'], betalist[[2]][x,'D_BC_dead'])) 
    
    # reassign neighbourhood income based on distribution above 
    D_new <- D_BC %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_triang.D0[i, 1:25] <- rb.D0
    pv_triang.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}

# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D08 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D0[,x],c(.025,.5,.975))))
sum.ppv.D08[4,] <- adj.pv[,'D_BC_alive']
sum.ppv.D08[5,] <- adj.lower[,'D_BC_alive']
sum.ppv.D08[6,] <- adj.upper[,'D_BC_alive']
row.names(sum.ppv.D08)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D08) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D18 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D1[,x],c(.025,.5,.975))))
sum.ppv.D18[4,] <- adj.pv[,'D_BC_dead']
sum.ppv.D18[5,] <- adj.lower[,'D_BC_dead']
sum.ppv.D18[6,] <- adj.upper[,'D_BC_dead']
row.names(sum.ppv.D18)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D18) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                           "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                           "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                           "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                           "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

# summarise output
# make sure there are no weird values 
output8 <- output %>% filter(se_q1!=0 & se_q1<100 |
                               se_q2!=0 & se_q2<100 |
                               se_q3!=0 & se_q3<100 |
                               se_q4!=0 & se_q4<100 )
# calculate quintiles 
q1_e<-quantile(output8$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output8$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output8$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output8$rr_q4_tot,c(.025,.5,.975))

# calculate quintiles without error
q1<-quantile(output8$rr_q1,c(.025,.5,.975))
q2<-quantile(output8$rr_q2,c(.025,.5,.975))
q3<-quantile(output8$rr_q3,c(.025,.5,.975))
q4<-quantile(output8$rr_q4,c(.025,.5,.975))

results8 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results8) <- c("2.5_adj","50_adj", "97.5_adj")

## COMPARE BC ##
# neighbourhood model 
obs_bc=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
              data    = D_BC,
              family  = poisson(link = "log"),
              id      = person_ID,
              corstr  = "exchangeable")

# save output to compare
results8$RR_obs_bc <- exp(coef(obs_bc))[2:5]
results8$lwr_obs_bc <- exp(confint.geeglm(obs_bc))[2:5,1]
results8$upr_obs_bc <- exp(confint.geeglm(obs_bc))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true_bc=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
               data    = D_BC,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")

# save output to compare
results8$RR_true_bc <- exp(coef(true_bc))[2:5]
results8$lwr_true_bc <- exp(confint.geeglm(true_bc))[2:5,1]
results8$upr_true_bc <- exp(confint.geeglm(true_bc))[2:5,2]

results8




# 10.QBA PR ----------------------------------------------------------

# set up data 
I=10
o=rep(0,I)
N <- 3851
D_PR <- D %>% filter(prov=='PR')
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}

pv_triang.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_triang.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_PR_alive'], betalist[[2]][x,'D_PR_alive'])) 
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_PR_dead'], betalist[[2]][x, 'D_PR_dead'])) 
    
    # reassign neighbourhood income based on distribution above 
    D_new <- D_PR %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_triang.D0[i, 1:25] <- rb.D0
    pv_triang.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}

# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D010 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D0[,x],c(.025,.5,.975))))
sum.ppv.D010[4,] <- adj.pv[,'D_PR_alive']
sum.ppv.D010[5,] <- adj.lower[,'D_PR_alive']
sum.ppv.D010[6,] <- adj.upper[,'D_PR_alive']
row.names(sum.ppv.D010)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D010) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                            "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                            "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                            "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D110 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D1[,x],c(.025,.5,.975))))
sum.ppv.D110[4,] <- adj.pv[,'D_PR_dead']
sum.ppv.D110[5,] <- adj.lower[,'D_PR_dead']
sum.ppv.D110[6,] <- adj.upper[,'D_PR_dead']
row.names(sum.ppv.D110)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D110) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                            "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                            "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                            "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

# summarise output
# make sure there are no weird values 
output10 <- output %>% filter(se_q1!=0 & se_q1<100 |
                                se_q2!=0 & se_q2<100 |
                                se_q3!=0 & se_q3<100 |
                                se_q4!=0 & se_q4<100 )
# calculate quintiles 
q1_e<-quantile(output10$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output10$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output10$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output10$rr_q4_tot,c(.025,.5,.975))

# calculate quintiles without error
q1<-quantile(output10$rr_q1,c(.025,.5,.975))
q2<-quantile(output10$rr_q2,c(.025,.5,.975))
q3<-quantile(output10$rr_q3,c(.025,.5,.975))
q4<-quantile(output10$rr_q4,c(.025,.5,.975))

results10 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results10) <- c("2.5_adj","50_adj", "97.5_adj")

## COMPARE ##
# neighbourhood model 
obs_pr=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
              data    = D_PR,
              family  = poisson(link = "log"),
              id      = person_ID,
              corstr  = "exchangeable")

# save output to compare
results10$RR_obs_pr <- exp(coef(obs_pr))[2:5]
results10$lwr_obs_pr <- exp(confint.geeglm(obs_pr))[2:5,1]
results10$upr_obs_pr <- exp(confint.geeglm(obs_pr))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true_pr=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
               data    = D_PR,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")

# save output to compare
results10$RR_true_pr <- exp(coef(true_pr))[2:5]
results10$lwr_true_pr <- exp(confint.geeglm(true_pr))[2:5,1]
results10$upr_true_pr <- exp(confint.geeglm(true_pr))[2:5,2]
results10





# 12.QBA - AT --------------------------------------------------------
I=10
o=rep(0,I)
N <- 2325
D_AT <- D %>% filter(prov=='AT')
output=data.frame(iter=1:I,rr_q1=o,rr_q2=o,rr_q3=o,rr_q4=o,
                  se_q1=o,se_q2=o,se_q3=o,se_q4=o,
                  rr_q1_tot=o,rr_q2_tot=o,rr_q3_tot=o,rr_q4_tot=o)

# dataset for sampled predictive values
for(i in 1:25) {                    
  assign(paste0("pv", i), o)
}

pv_triang.D0 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))
pv_triang.D1 <- data.frame(sapply(paste0("pv", 1:25), function(x) get(x)))

# loop 
for (i in 1:I) {
  
  tryCatch({
    
    # sample PV using beta distribution specified above   
    rb.D0 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_AT_alive'], betalist[[2]][x,'D_AT_alive'])) 
    rb.D1 <- sapply(1:25, function(x) rbeta(1, alphalist[[2]][x,'D_AT_dead'], betalist[[2]][x,'D_AT_dead'])) 
    
    # reassign neighbourhood income based on distribution above 
    D_new <- D_AT %>% mutate(e_new=case_when(
      neighb==1 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[1:5]),
      neighb==2 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[6:10]),
      neighb==3 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[11:15]),
      neighb==4 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0[16:20]),
      neighb==5 & dead_5yr==0 ~ sample(1:5, N, replace=TRUE, prob=rb.D0 [21:25]), 
      neighb==1 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[1:5]),
      neighb==2 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[6:10]),
      neighb==3 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[11:15]),
      neighb==4 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1[16:20]),
      neighb==5 & dead_5yr==1 ~ sample(1:5, N, replace=TRUE, prob=rb.D1 [21:25])
    ))
    
    D_new$neighb_1 <- ifelse(D_new$e_new==1, 1, 0)
    D_new$neighb_2 <- ifelse(D_new$e_new==2, 1, 0)
    D_new$neighb_3 <- ifelse(D_new$e_new==3, 1, 0)
    D_new$neighb_4 <- ifelse(D_new$e_new==4, 1, 0)
    
    res=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
               data    = D_new,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")
    
    coef=exp(coef(res))
    se=coef(summary(res))[,2]
    
    output$rr_q1[i]=coef[2]
    output$rr_q2[i]=coef[3]
    output$rr_q3[i]=coef[4]
    output$rr_q4[i]=coef[5]
    
    output$se_q1[i]=se[2]
    output$se_q2[i]=se[3]
    output$se_q3[i]=se[4]
    output$se_q4[i]=se[5]
    
    output$rr_q1_tot[i]=coef[2]+rnorm(1)*se[4]
    output$rr_q2_tot[i]=coef[3]+rnorm(1)*se[3]
    output$rr_q3_tot[i]=coef[4]+rnorm(1)*se[2]
    output$rr_q4_tot[i]=coef[5]+rnorm(1)*se[1]
    
    pv_triang.D0[i, 1:25] <- rb.D0
    pv_triang.D1[i, 1:25] <- rb.D1
    
    
  }
  
  , error = function(e) {cat("ERROR :",conditionMessage(e), "\n")})
  
  
}


# make sure sampled PVs match range for PVs in sesp dataframe
#D0
sum.ppv.D012 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D0[,x],c(.025,.5,.975))))
sum.ppv.D012[4,] <- adj.pv[,'D_AT_alive']
sum.ppv.D012[5,] <- adj.lower[,'D_AT_alive']
sum.ppv.D012[6,] <- adj.upper[,'D_AT_alive']
row.names(sum.ppv.D012)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D012) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                            "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                            "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                            "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")

#D1
sum.ppv.D112 <- data.frame(sapply(1:25, function(x) quantile(pv_triang.D1[,x],c(.025,.5,.975))))
sum.ppv.D112[4,] <- adj.pv[,'D_AT_dead']
sum.ppv.D112[5,] <- adj.lower[,'D_AT_dead']
sum.ppv.D112[6,] <- adj.upper[,'D_AT_dead']
row.names(sum.ppv.D112)[4:6] <- c("pv", "lower", "upper") 
colnames(sum.ppv.D112) <- c("Q1.1", "Q1.2", "Q1.3", "Q1.4", "Q1.5",
                            "Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5",
                            "Q3.1", "Q3.2", "Q3.3", "Q3.4", "Q3.5",
                            "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5",
                            "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5")


# summarise output
# make sure there are no weird values 
output12 <- output %>% filter(se_q1!=0 & se_q1<100 |
                                se_q2!=0 & se_q2<100 |
                                se_q3!=0 & se_q3<100 |
                                se_q4!=0 & se_q4<100 )
# calculate quintiles 
q1_e<-quantile(output12$rr_q1_tot,c(.025,.5,.975))
q2_e<-quantile(output12$rr_q2_tot,c(.025,.5,.975))
q3_e<-quantile(output12$rr_q3_tot,c(.025,.5,.975))
q4_e<-quantile(output12$rr_q4_tot,c(.025,.5,.975))

# calculate quintiles without error
q1<-quantile(output12$rr_q1,c(.025,.5,.975))
q2<-quantile(output12$rr_q2,c(.025,.5,.975))
q3<-quantile(output12$rr_q3,c(.025,.5,.975))
q4<-quantile(output12$rr_q4,c(.025,.5,.975))

results12 <- data.frame(rbind(q1,q2,q3,q4, q1_e,q2_e,q3_e,q4_e))
colnames(results12) <- c("2.5_adj","50_adj", "97.5_adj")

## COMPARE ##
# neighbourhood model 
obs_at=geeglm(formula = dead_5yr ~ neighb_1+neighb_2+neighb_3+neighb_4,
              data    = D_AT,
              family  = poisson(link = "log"),
              id      = person_ID,
              corstr  = "exchangeable")

# save output to compare
results12$RR_obs_at <- exp(coef(obs_at))[2:5]
results12$lwr_obs_at <- exp(confint.geeglm(obs_at))[2:5,1]
results12$upr_obs_at <- exp(confint.geeglm(obs_at))[2:5,2]

# modified poisson of true data - income income --> death within 5 years 
true_at=geeglm(formula = dead_5yr ~ ind_1+ind_2+ind_3+ind_4,
               data    = D_AT,
               family  = poisson(link = "log"),
               id      = person_ID,
               corstr  = "exchangeable")

# save output to compare
results12$RR_true_at <- exp(coef(true_at))[2:5]
results12$lwr_true_at <- exp(confint.geeglm(true_at))[2:5,1]
results12$upr_true_at <- exp(confint.geeglm(true_at))[2:5,2]
results12



