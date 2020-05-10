from normalization import normalize_corpus
from utils import build_feature_matrix
from bm25 import compute_corpus_term_idfs
from bm25 import compute_bm25_similarity
from semantic_similarity import sentence_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from csv import reader
import csv

CSVStudentAnswer=[""]
CSVCosineGrade=[]
CSVOntoGrade=[]


def cosineAlgorithm():
    print("\n")
    print("\n")
    print("************************ THE COSINE ALGORITHM *************************")
    print("\n")
    
    modelAnswer=""

    with open('ModelAnswer.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            for ele in row:
                modelAnswer=modelAnswer+ele

    studentAnswerTemp=[""]
    studentAnswerFinal=[""]
    temp=""
 
    with open('StudentAnswer.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            temp=""
            for ele in row:
                temp=temp+ele
            studentAnswerTemp[0]=temp
            studentAnswerFinal.extend(studentAnswerTemp)
            CSVStudentAnswer.extend(studentAnswerTemp)

    studentAnswerFinal.remove("")
    CSVStudentAnswer.remove("")
    
    print("The Modern Answer is :","\n",modelAnswer,"\n")

    m=1

    #Number of student answers should be even
    
    for i,j in zip(range(0,len(studentAnswerFinal),2), range(1,len(studentAnswerFinal),2)):
        
        documents = [modelAnswer, studentAnswerFinal[i], studentAnswerFinal[j]]

        # Create the Document Term Matrix
        count_vectorizer = CountVectorizer(stop_words='english')

        count_vectorizer = CountVectorizer()

        sparse_matrix = count_vectorizer.fit_transform(documents)

        # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
        doc_term_matrix = sparse_matrix.todense()
    
        df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(), index=['doc_trump', 'doc_election', 'doc_putin'])
    
        #print(cosine_similarity(df, df)*100)

        if(cosine_similarity(df, df)[0][1:2]*100<10.00):
            sim_score1=0    
        elif(10.00<=cosine_similarity(df, df)[0][1:2]*100<=20.00):
            sim_score1=1
        elif(20.00<cosine_similarity(df, df)[0][1:2]*100<=40.00):
            sim_score1=2
        elif(40.00<cosine_similarity(df, df)[0][1:2]*100<=60.00):
            sim_score1=3
        elif(60.00<cosine_similarity(df, df)[0][1:2]*100<=80.00):
            sim_score1=4
        elif(80.00<cosine_similarity(df, df)[0][1:2]*100<=100.00):
            sim_score1=5

        if(cosine_similarity(df, df)[0][2:3]*100<10.00):
            sim_score2=0
        elif(10.00<=cosine_similarity(df, df)[0][2:3]*100<=20.00):
            sim_score2=1
        elif(20.00<cosine_similarity(df, df)[0][2:3]*100<=40.00):
            sim_score2=2
        elif(40.00<cosine_similarity(df, df)[0][2:3]*100<=60.00):
            sim_score2=3
        elif(60.00<cosine_similarity(df, df)[0][2:3]*100<=80.00):
            sim_score2=4
        elif(80.00<cosine_similarity(df, df)[0][2:3]*100<=100.00):
            sim_score2=5
    
        print("Answer ",m," : ","\n",studentAnswerFinal[i],"\n")

        print("Similarity of Modern Answer and Answer 1 : ", cosine_similarity(df, df)[0][1:2]*100,"\n")

        print("The Grade for Answer 1 is : ", sim_score1, "\n")
        
        CSVCosineGrade.append(sim_score1)

        m=m+1
    
        print("Answer ",m," : ","\n",studentAnswerFinal[j],"\n")

        print("Similarity of Modern Answer and Answer 2 : ", cosine_similarity(df, df)[0][2:3]*100,"\n")

        print("The Grade for Answer 1 is : ", sim_score2, "\n")
        
        CSVCosineGrade.append(sim_score2)
    
        m=m+1
    
    
def run():
    print("\n")
    print("******************** THE ONTOLOGY ALGORITHM ************************")
    print("\n")
    
    modelAnswer=['']
    
    with open('ModelAnswer.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            for ele in row:
                modelAnswer.append(ele)
    
    modelAnswer.remove("")
    studentAnswerTemp=[""]
    studentAnswerFinal=[""]
    temp=""
 
    with open('StudentAnswer.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            temp=""
            for ele in row:
                temp=temp+ele
            studentAnswerTemp[0]=temp
            studentAnswerFinal.extend(studentAnswerTemp)

    studentAnswerFinal.remove("")
    
    # normalize answers
    norm_corpus = normalize_corpus(studentAnswerFinal, lemmatize=True)
    # normalize model_answer
    norm_model_answer =  normalize_corpus(modelAnswer, lemmatize=True)            
    vectorizer, corpus_features = build_feature_matrix(norm_corpus,feature_type='frequency')
    # extract features from model_answer
    model_answer_features = vectorizer.transform(norm_model_answer)
    doc_lengths = [len(doc.split()) for doc in norm_corpus]   
    avg_dl = np.average(doc_lengths) 
    corpus_term_idfs = compute_corpus_term_idfs(corpus_features, norm_corpus)
    for index, doc in enumerate(modelAnswer):
        doc_features = model_answer_features[index]
        bm25_scores = compute_bm25_similarity(doc_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=1.5, b=0.75)
        semantic_similarity_scores=[]
        for sentence in studentAnswerFinal:
            score=(sentence_similarity(sentence,modelAnswer[0])+sentence_similarity(modelAnswer[0],sentence))/2
            semantic_similarity_scores.append(score)
        print('Model Answer',':', doc)
        print('-'*40)
        doc_index=0
        for score_tuple in zip(semantic_similarity_scores,bm25_scores):
            sim_score=((score_tuple[0]*10)+score_tuple[1])/2
            if(sim_score<1):
                sim_score=0
            elif(1<=sim_score<=2):
                sim_score=1
            elif(2<sim_score<=4):
                sim_score=2
            elif(4<sim_score<=6):
                sim_score=3
            elif(6<sim_score<=8):
                sim_score=4
            elif(8<sim_score<=10):
                sim_score=5
            print('Ans num: {} Score: {}\nAnswer: {}'.format(doc_index+1, sim_score, studentAnswerFinal[doc_index]))
            CSVOntoGrade.append(sim_score)
            print('-'*40)
            doc_index=doc_index+1

run()
cosineAlgorithm()
pd.DataFrame({'Student-Answer': CSVStudentAnswer, 'Cosine-Grade': CSVCosineGrade, 'Ontology-Grade': CSVOntoGrade}).to_csv('GradeFile.csv', index=False)

'''
OUTPUT
runfile('D:/FINAL_YEAR/SEMESTER_08/STUDENT-GRADING-SYSTEM/DUPLICATE-COPY/Text-Similarity-master/execute.py', wdir='D:/FINAL_YEAR/SEMESTER_08/STUDENT-GRADING-SYSTEM/DUPLICATE-COPY/Text-Similarity-master')
Reloaded modules: normalization, utils, bm25, semantic_similarity


******************** THE ONTOLOGY ALGORITHM ************************


Model Answer : Computer science is the study of computers and computing concepts. It includes both hardware and software, as well as networking and the Internet. Programming concepts include functions, algorithms, and source code design. Computer science also covers compilers, operating systems, and software applications
----------------------------------------
Ans num: 1 Score: 3
Answer: a branch of science that deals with the theory of computation or the design of computers
----------------------------------------
Ans num: 2 Score: 5
Answer: Computer science, the study of computers and computing, including their theoretical and algorithmic foundations, hardware and software, and their uses for processing information.
----------------------------------------
Ans num: 3 Score: 5
Answer: Computer Science is the study of computers and computational systems. Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems; this includes their theory, design, development, and application.
----------------------------------------
Ans num: 4 Score: 5
Answer: Computer science is the study of computers, including computational theory, hardware and software design, algorithms, and the way humans interact with technology
----------------------------------------
Ans num: 5 Score: 5
Answer: the study of computers and algorithmic processes, including their principles, their hardware and software designs, their applications, and their impact on society.
----------------------------------------
Ans num: 6 Score: 4
Answer: Computer science is a very large subject with lots of applications. Computer scientists design new software, solve computing problems and develop different ways to use technology
----------------------------------------




************************ THE COSINE ALGORITHM *************************


The Modern Answer is : 
 Computer science is the study of computers and computing concepts. It includes both hardware and software, as well as networking and the Internet. Programming concepts include functions, algorithms, and source code design. Computer science also covers compilers, operating systems, and software applications 

Answer  1  :  
 a branch of science that deals with the theory of computation or the design of computers 

Similarity of Modern Answer and Answer 1 :  [26.66323911] 

The Grade for Answer 1 is :  2 

Answer  2  :  
 Computer science, the study of computers and computing, including their theoretical and algorithmic foundations, hardware and software, and their uses for processing information. 

Similarity of Modern Answer and Answer 2 :  [63.06628048] 

The Grade for Answer 1 is :  4 

Answer  3  :  
 Computer Science is the study of computers and computational systems. Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems; this includes their theory, design, development, and application. 

Similarity of Modern Answer and Answer 1 :  [66.4410597] 

The Grade for Answer 1 is :  4 

Answer  4  :  
 Computer science is the study of computers, including computational theory, hardware and software design, algorithms, and the way humans interact with technology 

Similarity of Modern Answer and Answer 2 :  [61.5547016] 

The Grade for Answer 1 is :  4 

Answer  5  :  
 the study of computers and algorithmic processes, including their principles, their hardware and software designs, their applications, and their impact on society. 

Similarity of Modern Answer and Answer 1 :  [44.11287733] 

The Grade for Answer 1 is :  3 

Answer  6  :  
 Computer science is a very large subject with lots of applications. Computer scientists design new software, solve computing problems and develop different ways to use technology 

Similarity of Modern Answer and Answer 2 :  [40.26936331] 

The Grade for Answer 1 is :  3
'''