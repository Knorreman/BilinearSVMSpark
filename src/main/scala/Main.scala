import model.{BilinearSVM, BilinearSVMModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("BilinearSVMTest")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // Generate synthetic data
    val numSamples = 100_000
    val numFeatures = 5 // Number of features in x1 and x2
    val data: RDD[(Double, Vector)] = generateSimilarDataSingleVector(sc, numSamples, numFeatures)

    val scaler = new StandardScaler(false, false)

    val splitData = data.randomSplit(Array(0.7, 0.3), 42)

    val training = splitData(0)
      .repartition(sc.defaultParallelism)
      .cache()

    val scalerModel = scaler.fit(training.map(_._2))
    val scalerModelBc = sc.broadcast(scalerModel)

    val trainingScaled = training.mapValues(v => scalerModelBc.value.transform(v))
      .map(tpl => LabeledPoint(tpl._1, tpl._2))
      .cache()

    val test = splitData(1)
      .mapValues(v => scalerModelBc.value.transform(v))
      .repartition(sc.defaultParallelism)
      .cache()

    val numIterations = 400
    val stepSize = 1e-4
    val regParam = 0.1

    trainingScaled.count()
    val t1 = System.currentTimeMillis()

    val bilinearSVM = new BilinearSVM()
      .setRegParam(regParam)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)

    val biSVMModel: BilinearSVMModel = bilinearSVM.run(trainingScaled)

    println(biSVMModel.weightMatrix.toString(100, 500))
    println(biSVMModel.intercept)

    val predictionsAndLabelsRaw = test.map { case (label, features) =>
      val prediction = biSVMModel.predict(features)
      (prediction, label)
    }

    val metrics = new BinaryClassificationMetrics(predictionsAndLabelsRaw)
    val precision = metrics.precisionByThreshold
    val recall = metrics.recallByThreshold

    val PRC = metrics.pr

    val f1Score = metrics.fMeasureByThreshold
    val max = f1Score.collect().maxBy(tpl => tpl._2)
    println(s"Threshold: ${max._1}, F-score: ${max._2}, Beta = 1")

    val auPRC = metrics.areaUnderPR
    println(s"Area under precision-recall curve = $auPRC")

    val thresholds = precision.map(_._1)

    val roc = metrics.roc

    val auROC = metrics.areaUnderROC
    println(s"Area under ROC = $auROC")

    val predictionsAndLabels = test.map { case (label, features) =>
      val prediction = biSVMModel.predict(features)

      val predictedLabel = if (prediction >= max._1) 1.0 else -1.0
      (predictedLabel, label)
    }.cache()

    val accuracy = predictionsAndLabels.filter { case (predicted, actual) => predicted == actual }.count().toDouble / test.count().toDouble
    println(s"Accuracy: $accuracy")

    val mcm = new MulticlassMetrics(predictionsAndLabels)
    println(s"accuracy: ${mcm.accuracy}")
    println(s"weightedRecall: ${mcm.weightedRecall}")
    println(s"weightedPrecision: ${mcm.weightedPrecision}")
    println(s"weightedFMeasure: ${mcm.weightedFMeasure}")
    println(s"weightedTruePositiveRate: ${mcm.weightedTruePositiveRate}")
    println(s"weightedFalsePositiveRate: ${mcm.weightedFalsePositiveRate}")

    val t = System.currentTimeMillis() - t1
    println("Spark runtime: " + t / 1.0e3)
    println("Testing saving and loading")

    val testPath: String = "model"

    biSVMModel.save(testPath)

    val loadedModel = BilinearSVMModel.load(testPath)

    println(loadedModel.weightMatrix)
    println(loadedModel.intercept)

    sc.stop()
  }

  private def generateSimilarData(sc: SparkContext, numSamples: Int, numFeatures: Int): RDD[(Double, Vector)] = {
    val rnd = new scala.util.Random(42)
    import breeze.stats.distributions._
    import Rand.VariableSeed.randBasis
    randBasis.generator.setSeed(42)
    val exp = Exponential(1.0)
    val gamma = Gamma(1, 2)
    val beta = Beta(1, 3)
    val poi = Poisson(2)
    val poi2 = Poisson(1)

    // Generate positive samples
    val positiveSamples = (1 to numSamples / 2).map { _ =>
      val expArr = poi.sample(numFeatures).toArray.map(_.toDouble).map(_ - 0)
      val x1 = Vectors.dense(expArr)
      val x2 = Vectors.dense(expArr)
      val features = Vectors.dense(x1.toArray ++ x2.toArray)
      (1.0, features)
    }

    // Generate negative samples
    val negativeSamples = (1 to numSamples / 2).map { _ =>
      val gamArr = poi2.sample(numFeatures).toArray.map(_.toDouble).map(_ - 0)
      val x1 = Vectors.dense(gamArr)
      val x2 = Vectors.dense(gamArr)
      val features = Vectors.dense(x1.toArray ++ x2.toArray)
      (-1.0, features)
    }

    sc.parallelize(positiveSamples ++ negativeSamples, sc.defaultParallelism)
  }

  private def generateSimilarDataSingleVector(sc: SparkContext, numSamples: Int, numFeatures: Int): RDD[(Double, Vector)] = {
    val rnd = new scala.util.Random(42)
    import breeze.stats.distributions._
    import Rand.VariableSeed.randBasis
    randBasis.generator.setSeed(42)
    val exp = Exponential(1.0)
    val gamma = Gamma(1, 2)
    val beta = Beta(1, 3)
    val poi = Poisson(2)
    val poi2 = Poisson(1)

    // Generate positive samples
    val positiveSamples = (1 to numSamples / 2).map { _ =>
      val expArr = beta.sample(numFeatures).toArray.map(_.toDouble).map(_ - 0)
      val features = Vectors.dense(expArr)
      (1.0, features)
    }

    // Generate negative samples
    val negativeSamples = (1 to numSamples / 2).map { _ =>
      val gamArr = gamma.sample(numFeatures).toArray.map(_.toDouble).map(_ - 0)
      val features = Vectors.dense(gamArr)
      (-1.0, features)
    }

    sc.parallelize(positiveSamples ++ negativeSamples, sc.defaultParallelism)
  }

}