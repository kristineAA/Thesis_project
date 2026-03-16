
# set wd
setwd("C:/Users/dillonm1/Dropbox (Aalto)/Ellie/PhD work/Research/Blood 2020/Platelet 2020/Hospital_demand_data/demand_data")

# read in data
hosp_1_18_19<- read.csv("hosp1_18_19_demand.csv", sep = ",", header = TRUE)
hosp_2_18_19<- read.csv("hosp2_18_19_demand.csv", sep = ",", header = TRUE)
hosp_3_18_19<- read.csv("hosp3_18_19_demand.csv", sep = ",", header = TRUE)
hosp_4_18_19<- read.csv("hosp4_18_19_demand.csv", sep = ",", header = TRUE)
hosp_5_18_19<- read.csv("hosp5_18_19_demand.csv", sep = ",", header = TRUE)
hosp_6_18_19<- read.csv("hosp6_18_19_demand.csv", sep = ",", header = TRUE)
hosp_med_18_19<- read.csv("med_18_19_demand.csv", sep = ",", header = TRUE)
hosp_small_18_19<- read.csv("small_18_19_demand.csv", sep = ",", header = TRUE)

# sum units delivered to each hospital
num_del_h1<- sum(hosp_1_18_19$delivery)
num_del_h2<- sum(hosp_2_18_19$delivery)
num_del_h3<- sum(hosp_3_18_19$delivery)
num_del_h4<- sum(hosp_4_18_19$delivery)
num_del_h5<- sum(hosp_5_18_19$delivery)
num_del_h6<- sum(hosp_6_18_19$delivery)
num_del_med<- sum(hosp_med_18_19$delivery)
num_del_small<- sum(hosp_small_18_19$delivery)

num_del_total<- sum(num_del_h1,num_del_h2,num_del_h3,num_del_h4,num_del_h5,num_del_h6,num_del_med,num_del_small)

# calc proportion of demand

demand_h1<- num_del_h1/num_del_total
demand_h2<- num_del_h2/num_del_total
demand_h3<- num_del_h3/num_del_total
demand_h4<- num_del_h4/num_del_total
demand_h5<- num_del_h5/num_del_total
demand_h6<- num_del_h6/num_del_total
demand_med<- num_del_med/num_del_total
demand_small<- num_del_small/num_del_total

# large
demand_large<- sum(num_del_h1,num_del_h2,num_del_h3,num_del_h4,num_del_h5,num_del_h6)/num_del_total

sum(demand_large,demand_med,demand_small)


# count number of medium and small hosps

# full data file with size of hospital
all_data<- read.csv("20200611DailyCountsOfPlateletReturnsAndDeliveries.txt", header=TRUE, sep=";")

# subset dfs by size
med_hosp<-subset(all_data,size=="medium")
# drop unused levels
med_hosp[] <- lapply(med_hosp, function(x) if(is.factor(x)) factor(x) else x)

small_hosp<-subset(all_data,size=="small")
# drop unused levels
small_hosp[] <- lapply(small_hosp, function(x) if(is.factor(x)) factor(x) else x)

## how many med hosps? 19
split_list_med <- split(med_hosp,med_hosp$name)

## how many small hosps? 37
split_list_small <- split(small_hosp,small_hosp$name)




