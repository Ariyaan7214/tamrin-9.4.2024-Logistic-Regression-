# tamrin-9.4.2024-Logistic-Regression-
(Logistic Regression) tamrin 9.4.2024 github
!pip install pyspark
from pyspark.sql import SparkSession as ss
spark = ss.builder.appName('tamrinlogreg').getOrCreate()
from google.colab import drive
drive.mount('/content/drive')
df=spark.read.csv('/content/drive/MyDrive/heart.csv',header=True,inferSchema=True)
df.show()
df.columns
from pyspark.ml.feature import VectorAssembler as va
asmbl=va(inputCols=['age',
 'sex',
 'cp',
 'trestbps',
 'chol',
 'fbs',
 'restecg',
 'thalach',
 'exang',
 'oldpeak',
 'slope',
 'ca',
 'thal',
 'target'],outputCol='features')
 output = asmbl.transform(df)
 output.select('features').show()
 finaldata=output.select('features','target')
 finaldata.show()
 traintarget,testtarget=finaldata.randomSplit([0.7,0.3])
 traintarget.describe().show()
 testtarget.describe().show()
 from pyspark.ml.classification import LogisticRegression as lr
lrtarget=lr(labelCol='target')
fittedtargetmodel=lrtarget.fit(traintarget)
predandlabels=fittedtargetmodel.evaluate(testtarget)
predandlabels.predictions.show()
from pyspark.ml.evaluation import BinaryClassificationEvaluator as bce
targeteval=bce(rawPredictionCol='prediction',labelCol='target')
result=targeteval.evaluate(predandlabels.predictions)
result
