package model

import model.gradient.BilinearSVMGradient
import model.updater.BilinearSVMUpdater
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.optimization.{GradientDescent, L1Updater, LBFGS, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class BilinearSVM(private var numFeatures: Int = -1,
                   private var numIterations: Int = 100,
                   private var stepSize: Double = 0.01,
                   private var regParam: Double = 0.1) extends Serializable {

  def setNumIterations(n: Int): BilinearSVM = {
    this.numIterations = n
    this
  }

  def setStepSize(d: Double): BilinearSVM = {
    this.stepSize = d
    this
  }

  def setRegParam(p: Double): BilinearSVM = {
    this.regParam = p
    this
  }

  def setNumFeatures(n: Int): BilinearSVM = {
    this.numFeatures = n
    this
  }

  def run(data: RDD[LabeledPoint]): BilinearSVMModel = {
    if (numFeatures < 0) {
      numFeatures = data.map(_.features.size).first()
    }

    BilinearSVM.trainBilinearSVMLBFGS(data,
      numFeatures,
      numIterations,
      regParam)
  }

  def runSGD(data: RDD[LabeledPoint]): BilinearSVMModel = {
    if (numFeatures < 0) {
      numFeatures = data.map(_.features.size).first()
    }

    BilinearSVM.trainBilinearSVMSGD(data,
      numFeatures,
      numIterations,
      stepSize,
      regParam)
  }
}

object BilinearSVM {

  private def trainBilinearSVMLBFGS(data: RDD[LabeledPoint],
                                    numFeatures: Int,
                                    numIterations: Int,
                                    regParam: Double
                           ): BilinearSVMModel = {
    val gradient = new BilinearSVMGradient(numFeatures)

    val initialWeights = Vectors.zeros(numFeatures * numFeatures + 1)

    val dataTuple = data.map(lp => (lp.label, lp.features))
    val (modelWeights, losses) = LBFGS.runLBFGS(dataTuple,
      gradient = gradient,
      updater = new BilinearSVMUpdater(),
      10,
      1e-9,
      maxNumIterations = numIterations,
      regParam = regParam,
      initialWeights = initialWeights)
    println(s"Converged after ${losses.length} iterations")
    val aArray = modelWeights.toArray.slice(0, numFeatures * numFeatures)
    val b = modelWeights(modelWeights.size - 1)
    val A = new DenseMatrix(numFeatures, numFeatures, aArray)
    new BilinearSVMModel(A, b)
  }

  private def trainBilinearSVMSGD(data: RDD[LabeledPoint],
                                  numFeatures: Int,
                                  numIterations: Int,
                                  stepSize: Double,
                                  regParam: Double
                      ): BilinearSVMModel = {
    val gradient = new BilinearSVMGradient(numFeatures)

    val initialWeights = Vectors.zeros(numFeatures * numFeatures + 1)
    val dataTuple = data.map(lp => (lp.label, lp.features))
    val (modelWeights, losses) = GradientDescent.runMiniBatchSGD(dataTuple,
      gradient,
      new BilinearSVMUpdater(),
      stepSize,
      numIterations,
      regParam,
      1.0,
      initialWeights,
      1e-9)
    println(s"Converged after ${losses.length} iterations")

    val aArray = modelWeights.toArray.slice(0, numFeatures * numFeatures)
    val b = modelWeights(modelWeights.size - 1)
    val A = new DenseMatrix(numFeatures, numFeatures, aArray)
    new BilinearSVMModel(A, b)
  }

}