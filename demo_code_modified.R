
#demo data input
item_name = c('apple', 'orange', 'apple', 'apple', 'orange', 'orange')
#date comes in the below format
date = c('01/01/16','02/01/16','12/25/15','01/15/16','04/07/16','03/25/16')
count = c(2,6,9,2,3,4)
mydata = data.frame(item_name, date, count)
# convertion of date entry to date format
mydata$date1 = as.Date(mydata$date, format = '%m/%d/%y') # I have changed the "Y" of format to "y"

# I want to calculate the total number of item which falls into a range of date

table1 = function(s,e){
  st = as.Date(s, format = "%m/%d/%y")
  ed = as.Date(e, format = '%m/%d/%y')
  mydata_n = mydata[mydata$date1 > st & mydata$date1 < ed, ] # I inserted this line and removed a few
  result = aggregate(count ~ item_name, sum, data = mydata_n)
  
  return (result)
} 
    
# test
table1(s = '12/31/15', e = '04/01/16')  
f1=as.data.frame(table1(s = '12/31/15', e = '04/01/16')  )

> table1(s = '12/31/15', e = '04/01/16')  
  item_name count
1     apple     4
2    orange    10
> 


