AUTOMATIC GRADING SYSTEM FOR ANSWER USING ONTOLOGY AND COSINE METHOD/ALGORITHM

--> PRE-REQUISITE LIBRARIES

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

--> MODERN/MODEL/TEACHER ANSWER

The teacher answer is put into .csv file as shown below. This is our first Input to code


![modern-answer](https://user-images.githubusercontent.com/34189979/81499372-c564b180-92e8-11ea-9b84-01068abf9abd.PNG)



--> STUDENTS ANSWERS

The student answers is put into .csv file rowwise (each row represents new student answer) as shown below. This is our second Input to code


![student-answer](https://user-images.githubusercontent.com/34189979/81499409-1b395980-92e9-11ea-939c-5dfb514d1e03.PNG)



--> FINAL CSV FILE WITH STUDENT ANSWERS ALONG WITH THEIR GRADES USING ONTOLOGY AND COSINE METHOD



![final-ontology](https://user-images.githubusercontent.com/34189979/81499439-62bfe580-92e9-11ea-96d6-d1abae721599.PNG)



HOW TO RUN THE CODE


Only run the execute.py file
