import re
 
# reading given tsv file
with open("train.tsv", 'r') as myfile:  
  with open("train.csv", 'w') as csv_file:
    for line in myfile:
       
      # Replace every tab with comma
      fileContent = line.split('\t')
      new_line = '"'+fileContent[0]+'"'+','+ '"'+fileContent[1]+'"'+','+ fileContent[2]

      # Writing into csv file
      csv_file.write(new_line)
 
# output
print("Successfully made csv file")