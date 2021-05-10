import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame,Row}
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.StandardScaler
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, mean, stddev}

object Main {
def main(args:Array[String]){
val spark: SparkSession = SparkSession.builder().appName("Main").getOrCreate()
val sc = spark.sparkContext

import spark.implicits._

//-------------------- CONSTANTS -----------------------------------------

// val inputFile = "hdfs:///user/ss13337/project/data0-4.csv"
val inputFile = args(0)
val tempFile = "/user/ss13337/project/temp/pca_op"

//-------------------- DATA LOADING  -------------------------------------

var data = spark.read.option("header","true").option("inferSchema","true").format("csv").load(inputFile)


//---------------------REMOVE OUTLIERS--------------------------------

val outlier_strength:Int = 3
var cols = data.columns.takeRight(1)
for (c <- cols) {
	val stats = data.agg(mean(c).as("mean"), stddev(c).as("stddev")).withColumn("UpperLimit", col("mean") + col("stddev") * outlier_strength)
	data = data.filter(data(c) < stats.first().get(2))
}
val meanCol: Double = data.agg(mean("4")).first().getDouble(0)

//-------------------- HANDLE NaN VALUES  -----------------------

//Replaces the NaN values with mean of that column
cols = data.columns.dropRight(1)
val imputedModel = new Imputer().setInputCols(cols).setOutputCols(cols).fit(data)
data = imputedModel.transform(data)


//---------------------INTO LIBSVM FORMAT---------------------------------

// Transform data into Label and Vector of Features for PCA
val vec = data.map{ row =>
	new LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(0), row.getDouble(1), row.getDouble(2), row.getDouble(3)))}.rdd.cache()

//----------------------TRANSFORM DATA---------------------------------

// Transform data into Label and Vector of Features
val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
data = vectorAssembler.transform(data)


//--------------------- LR MODEL PERFORMANCE WITHOUT PCA -----------------------------

// Scaling
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
val scalerModel = scaler.fit(data)
data = scalerModel.transform(data)

// ML splits
val Array(trainingData1, testData1) = data.randomSplit(Array(0.8, 0.2))

// LinearRegression
val lir = new LinearRegression().setLabelCol("4").setFeaturesCol("scaledFeatures").setMaxIter(100).setRegParam(0.1).setElasticNetParam(1.0)
val model = lir.fit(trainingData1)
val pred = model.evaluate(testData1)

// Evaluation
val evaluator = new RegressionEvaluator().setLabelCol("4").setPredictionCol("prediction").setMetricName("rmse")
val rmse1 = evaluator.evaluate(pred.predictions)


//--------------------- PCA PROCEDURE -------------------------------------

// PCA fit
val pca = new PCA(4).fit(vec.map(_.features))
val pcaData = vec.map(p => p.copy(features = pca.transform(p.features)))

// Load data to HDFS
MLUtils.saveAsLibSVMFile(pcaData.coalesce(1), tempFile)
//Unload data from HDFS
data = spark.read.format("libsvm").load(tempFile)

//--------------------- LR MODEL PERFORMANCE WITH PCA -----------------------------

// Scaling
val scaler2 = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
val scalerModel2 = scaler2.fit(data)
data = scalerModel2.transform(data)

// ML splits
val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

// LinearRegression
val lir2 = new LinearRegression().setFeaturesCol("scaledFeatures").setMaxIter(1000).setRegParam(0.01).setElasticNetParam(1.0)
val model2 = lir2.fit(trainingData)
val pred2 = model2.evaluate(testData)

// Evaluation
val evaluator2 = new RegressionEvaluator().setPredictionCol("prediction").setMetricName("rmse")
val rmse2 = evaluator2.evaluate(pred2.predictions)

//--------------------- GBT MODEL PERFORMANCE WITH PCA -----------------------------

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Train a GBT model.
val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxIter(10)

// Chain indexer and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(featureIndexer, gbt))

// Train model. This also runs the indexer.
val model3 = pipeline.fit(trainingData)

// Make predictions.
val predictions = model3.transform(testData)

// Select (prediction, true label) and compute test error.
val evaluator3 = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse3 = evaluator3.evaluate(predictions)


//----------------------DISPLAY OUTPUT AND DELETE THE TEMP LIBSVM FILE-------------------------

println("Linear Regression: Normalized Root Mean Squared Error = " + rmse1/meanCol)
println("Linear Regression with PCA: Normalized Root Mean Squared Error = " + rmse2/meanCol)
println("Gradient Boosted Tree with PCA: Normalized Root Mean Squared Error = " + rmse3/meanCol)

val fs = FileSystem.get(sc.hadoopConfiguration)
val outPutPath = new Path(tempFile)

if (fs.exists(outPutPath))
  fs.delete(outPutPath, true)

}
}
