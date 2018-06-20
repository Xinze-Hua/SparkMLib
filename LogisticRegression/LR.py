from pyspark import SparkConf, SparkContext
from pyspark.ml import pipeline
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel, BinaryLogisticRegressionSummary
from pyspark.sql.session import SparkSession
from pyspark.sql import Row, functions
 
spark = SparkSession.builder.master("local").appName("LR").getOrCreate()
 
 
def f(x):
    rel = {}
    rel['feature'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
    rel['label'] = str(x[4])
    return rel
 
 
def LRClassify():
    IrisData = spark.sparkContext.textFile('file:///home/unbroken/MyFiles/Work/Programming/Spark/LogisticRegression/Iris.txt')\
    .map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    labelIndexer = StringIndexer(inputCol='label', outputCol='indexLabel').fit(IrisData)
    featureIndexer = VectorIndexer(inputCol='feature', outputCol='indexFeatures').fit(IrisData)
    trainData, TestData = IrisData.randomSplit([0.7, 0.3])
    lr = LogisticRegression().setLabelCol('indexLabel').setFeaturesCol("indexFeatures").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictionLabel').setLabels(labelIndexer.labels)
    lrPipeline = Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])
    lrPipelineModel = lrPipeline.fit(trainData)
    lrPrediction = lrPipelineModel.transform(TestData)
    print(lrPrediction.show())
    evaluator = MulticlassClassificationEvaluator().setLabelCol("indexLabel").setPredictionCol("prediction")
    lrAccuracy = evaluator.evaluate(lrPrediction)
    print(lrAccuracy)
    
  
LRClassify()


