
# coding: utf-8

# In[3]:


from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
import logging


# In[4]:


conf=SparkConf().setAppName('bicycle')
sc=SparkContext(conf=conf)


# In[5]:


sql=SQLContext(sc)


# # Reading train datasets

# In[6]:


train=sql.read.options(inferschema=True,header=True).csv('train.csv')
train.show(2)


# # Exploratory data analysis of the train dataset

# In[6]:


train.printSchema()


# In[6]:


train.describe().show(truncate=False)


# In[7]:


train.dtypes


# In[ ]:


# find out how many blanks are available in each columns (only int data types)


# In[7]:


b=[(x,train.filter(col(x).isNull()).count())for x in train.columns]
b


# In[8]:


# method to prune the dataset to only numeric variables
che=[x[0] for x in train.dtypes if x[1].startswith('int')]
che


# # split the date time into month, day and year

# In[9]:


train1=train.select(year('datetime').alias('dt_year'),month('datetime').alias('dt_month'),dayofmonth('datetime').alias('dt_day'),
                   hour('datetime').alias('dt_hour'),
                   'season',
                   'holiday',
                   'workingday',
                   'weather',
                   'temp',
                   'atemp',
                   'humidity',
                   'windspeed',
                   'casual',
                   'registered',
                   'count')
train1.show(2)


# In[ ]:


# rename column count to label


# In[10]:


train1=train1.withColumnRenamed('count','label')
train1.show(2)


# # Deploying model

# In[11]:


#use vector assembler method to create feature vector
from pyspark.ml.feature import VectorAssembler


# In[12]:


v=VectorAssembler().setInputCols(['dt_year','dt_month','dt_hour','season',
                                      'holiday',
                                      'workingday',
                                      'weather',
                                      'temp',
                                      'atemp',
                                      'humidity',
                                      'windspeed',
                                      'casual',
                                      'registered',
                                     ]).setOutputCol('features')


# In[13]:


train4=v.transform(train1)
train4.show(2)


# In[14]:


train5=train4.select('label','features')
train5.show(2)


# In[15]:


# split the data into train and test
train003,test003=train5.randomSplit([0.7,0.3],seed=4000)
print('Total records in train set is ' +  str(train003.count()))
print('Total records in test set is ' +  str(test003.count()))


# In[16]:


# Linear regression
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
model_lr=lr.fit(train003)
train6=model_lr.transform(train003)
print("Coefficients: " + str(model_lr.coefficients))
print("Intercept: " + str(model_lr.intercept))


# In[17]:


trainingSummary = model_lr.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# # Applying model to the test dataset

# In[18]:


lr_predictions = model_lr.transform(test003)
lr_predictions.select("prediction","label","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="label",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))


# In[23]:


test_result = model_lr.evaluate(test003)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# # Gradiant boosted trees - https://docs.databricks.com/spark/latest/mllib/decision-trees.html
# 

# In[24]:


# read the file
df=sql.read.options(inferschema=True,header=True).csv('train.csv')


# In[25]:


df.show(2)


# In[26]:


# convert datetime into date, time, hour
dfa=df.select(year('datetime').alias('year'),month('datetime').alias('month'),dayofmonth('datetime').alias('day'),hour('datetime').alias('hour'),'season','holiday','workingday','weather',
             'temp','atemp','humidity','windspeed','casual','registered','count')


# In[27]:


# split the dataset into train and test

train,test=dfa.randomSplit([0.7,0.3],seed=4000)


# In[28]:


# prepare the dataset

col=dfa.columns
col.remove('count')
from pyspark.ml.feature import VectorAssembler,VectorIndexer

va=VectorAssembler(inputCols=col,outputCol='raw_feature')
vi=VectorIndexer(inputCol='raw_feature',outputCol='features',maxCategories=6)


# In[29]:


from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor


# In[30]:


# set up gbt
gbt=GBTRegressor(labelCol='count')


# In[32]:


# set up the parameters
grid=ParamGridBuilder().addGrid(gbt.maxDepth,[1,10]).addGrid(gbt.maxIter,[1,100]).build()


# In[33]:


evaluator=RegressionEvaluator(metricName='rmse',labelCol=gbt.getLabelCol(),predictionCol=gbt.getPredictionCol())


# In[34]:


cv=CrossValidator(estimator=gbt,estimatorParamMaps=grid,evaluator=evaluator)


# In[35]:


# create pipeline
from pyspark.ml import Pipeline
pipeline=Pipeline(stages=[va,vi,cv])


# In[36]:


model=pipeline.fit(train)


# In[35]:


prediction=model.transform(test)


# In[39]:


prediction.select('features','prediction').show()


# # persist the model

# In[42]:


from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel


# In[19]:


model_lr.write().save('home/edureka_37986')

