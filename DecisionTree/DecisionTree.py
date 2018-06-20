import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.feature import VectorIndexer, IndexToString, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('DecisionTree').getOrCreate()

def f(x):
    rel = {}
    rel['feature'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    rel['label'] = str(x[4])
    return rel

def DecisionTree():
    IrisData = spark.sparkContext.textFile("file:///home/unbroken/MyFiles/Work/Programming/Spark/DecisionTree/Iris.txt")\
    .map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()
    IrisData.createOrReplaceTempView("iris")
    df = spark.sql("select * from iris")
    labelIndexer = StringIndexer(inputCol='label', outputCol='labelIndex').fit(IrisData)
    featureIndexer = VectorIndexer(inputCol='feature', outputCol='indexFeature').setMaxCategories(4).fit(IrisData)
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictionLabel').setLabels(labelIndexer.labels)
    trainningData, testingData = IrisData.randomSplit([0.7, 0.3])
    dtClassifier = DecisionTreeClassifier().setLabelCol('labelIndex').setFeaturesCol('indexFeature')
    pipelineClassifier = Pipeline().setStages([labelIndexer, featureIndexer, dtClassifier, labelConverter])
    modelClassifier = pipelineClassifier.fit(trainningData)
    prediction = modelClassifier.transform(testingData)
    print(prediction.show())
    
    evaluator = MulticlassClassificationEvaluator().setLabelCol('labelIndex').setPredictionCol('prediction').setMetricName("accuracy")
    accuracy = evaluator.evaluate(prediction)
    print(accuracy)
    
    treeModelClassifier = modelClassifier.stages[2]
    print("Learned classification tree model:\n" + str(treeModelClassifier.toDebugString))

DecisionTree()