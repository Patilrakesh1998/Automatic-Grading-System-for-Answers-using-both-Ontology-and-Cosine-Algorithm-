'''
import csv

with open('temp.csv','w',newline='') as f:
    
    thewriter = csv.writer(f)
    thewriter.writerow(['Student-Answers','CosineGrade','OntologyGrade'])
    studentanswer=['aaa','bbb','ccc','ddd','fff','ggg']
    gradecosine=[2,6,4,1,5,3]
    gradeonto=[4,6,3,2,5,9]
    for i in range(6):
        thewriter.writerow([studentanswer[i],gradecosine[i],gradeonto[i]])
    
    studentanswer=['aaa','bbb','ccc','ddd','fff','ggg']
    gradecosine=[2,6,4,1,5,3]
    gradeonto=[4,6,3,2,5,9] 

    fieldnames=['Student-Answers','CosineGrade','OntologyGrade']
    thewriter=csv.DictWriter(f,fieldnames=fieldnames)
    thewriter.writeheader()
    
    for i in range(6):
        thewriter.writerow({'Student-Answers':studentanswer[i], 'CosineGrade':gradecosine[i]})
    
    for i in range(6):
        thewriter.writerow({'OntologyGrade':gradeonto[i]})
    '''
import csv
import pandas as pd
studentanswer=['aaa','bbb','ccc','ddd','fff','ggg']
gradecosine=[2,6,4,1,5,3]
gradeonto=[4,6,3,2,5,9]
pd.DataFrame({'A': studentanswer, 'B': gradecosine, 'C': gradeonto}).to_csv('file.csv', index=False)