#!/usr/bin/env python
# coding: utf-8

# In[212]:


from pyspark.sql.session import SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.types import *
from pyspark.sql import Window

import math
import re


# In[11]:


spark = SparkSession.builder.appName("word_count_segments").getOrCreate()


# In[12]:


import os

os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/local/softwares/anaconda3/bin/python3'
os.environ["PYSPARK_PYTHON"] = '/usr/local/softwares/anaconda3/bin/python3'

spark.stop()
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


# # Analyses for words

# In[243]:


# Reading the input data set 
words_rdd = sc.textFile("inputs/sample-a.txt")   

# Breaking text to words in different rows
# As there are many break chara
words_rdd2 = words_rdd.flatMap(lambda row: row.split("_"))
words_rdd3 = words_rdd2.flatMap(lambda row: row.split(" "))
words_rdd4 = words_rdd3.flatMap(lambda row: row.split(","))
words_rdd5 = words_rdd4.flatMap(lambda row: row.split("."))
words_rdd6 = words_rdd5.flatMap(lambda row: row.split(":"))
words_rdd7 = words_rdd6.flatMap(lambda row: row.split("{"))
words_rdd8 = words_rdd7.flatMap(lambda row: row.split("}"))
words_rdd9 = words_rdd8.flatMap(lambda row: row.split("("))
words_rdd10 = words_rdd9.flatMap(lambda row: row.split(")"))
words_rdd11 = words_rdd10.flatMap(lambda row: row.split(";"))
words_rdd12 = words_rdd11.flatMap(lambda row: row.split("["))
words_rdd13 = words_rdd12.flatMap(lambda row: row.split("-"))
words_rdd14 = words_rdd13.flatMap(lambda row: row.split('"'))
words_rdd15 = words_rdd14.flatMap(lambda row: row.split(']'))
words_rdd16 = words_rdd15.flatMap(lambda row: row.split('?'))
words_rdd17 = words_rdd16.flatMap(lambda row: row.split('!'))


# Removing break words
words_rdd_final = words_rdd17.filter(lambda x: x.strip() not in [''])                          .filter(lambda x: not x.isdigit())                          .filter(lambda x: re.match("^[A-Za-z]*$", x))                          .map(lambda x: (str(x.lower())))      

total_count = words_rdd_final.count()
distinct_count = words_rdd_final.distinct().count()

print("Total Number of words: ", total_count)
print("Total Number of distinct words: ", distinct_count)

popular_threshold = math.ceil(float(distinct_count*0.05))
common_threshold_lower = math.floor(float(distinct_count*0.475))
common_threshold_upper = math.ceil(float(distinct_count*0.525))
rare_threshold = math.floor(float(distinct_count*0.95))

print("Popular threshold: ", popular_threshold)
print("Common threshold (lower): ", common_threshold_lower)
print("Common threshold (upper): ", common_threshold_upper)
print("Rare threshold: ", rare_threshold)


# Converting RDD to Dataframe
schema = StructType([
                StructField("word", StringType())
            ])

data_df = words_rdd_final.map(lambda x: Row(x))                          .toDF(schema)


# Aggregating data
data_aggr = data_df.groupBy("word")                    .count()

# Forming window and calculating rank
win = Window.orderBy(fn.desc("count"), "word")
data_rank = data_aggr.withColumn("rank", fn.rank().over(win))


# Getting different sections of dataframe
popular_df = data_rank.where(fn.col("rank") <= popular_threshold)                       .select("rank", "word", "count")

common_df = data_rank.where((fn.col("rank") >= common_threshold_lower) & 
                            (fn.col("rank") <= common_threshold_upper)) \
                      .select("rank", "word", "count")

rare_df = data_rank.where(fn.col("rank") >= rare_threshold)                    .select("rank", "word", "count") 

# Displaying output
print("\nPopular words: ")
popular_df.show()

print("Common words: ")
common_df.show()

print("Rare words: ")
rare_df.show()


# # Analyses of Letter

# In[244]:


# getting letters

letters_rdd = words_rdd.flatMap(lambda row: list(row))                        .filter(lambda x: re.match("^[A-Za-z]*$", x))                        .map(lambda x: x.lower()) 

distinct_count = letters_rdd.distinct().count()
popular_threshold = math.ceil(float(distinct_count*0.05))
common_threshold_lower = math.floor(float(distinct_count*0.475))
common_threshold_upper = math.ceil(float(distinct_count*0.525))
rare_threshold = math.floor(float(distinct_count*0.95))

print("Total number of distinct letters", letters_rdd.distinct().count())
print("Popular threshold Letter: ", popular_threshold)
print("Common threshold Letter (lower): ", common_threshold_lower)
print("Common threshold Letter (upper): ", common_threshold_upper)
print("Rare threshold Letter: ", rare_threshold)

# Converting RDD to Dataframe
schema = StructType([
                StructField("letter", StringType())
            ])

data_df = letters_rdd.map(lambda x: Row(x))                      .toDF(schema)


# Aggregating data
data_aggr = data_df.groupBy("letter")                    .count()

# Forming window and calculating rank
win = Window.orderBy(fn.desc("count"), "letter")
data_rank = data_aggr.withColumn("rank", fn.rank().over(win))


# Getting different sections of dataframe
popular_df = data_rank.where(fn.col("rank") <= popular_threshold)                       .select("rank", "letter", "count")

common_df = data_rank.where((fn.col("rank") >= common_threshold_lower) & 
                            (fn.col("rank") <= common_threshold_upper)) \
                      .select("rank", "letter", "count")

rare_df = data_rank.where(fn.col("rank") >= rare_threshold)                    .select("rank", "letter", "count") 

# Displaying output
print("\nPopular letter: ")
popular_df.show()

print("Common letter: ")
common_df.show()

print("Rare letter: ")
rare_df.show()

