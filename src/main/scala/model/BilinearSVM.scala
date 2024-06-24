package model

import model.gradient.BilinearSVMGradient
import model.updater.BilinearSVMUpdater
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.optimization.{GradientDescent, LBFGS}
import org.apache.spark.rdd.RDD

class BilinearSVM(var numFeatures: Int = -1,
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

  def run(data: RDD[(Double, Vector)]): BilinearSVMModel = {
    BilinearSVM.trainBilinearSVMLBFGS(data,
      numFeatures,
      numIterations,
      regParam)
  }

  def runSGD(data: RDD[(Double, Vector)]): BilinearSVMModel = {
    BilinearSVM.trainBilinearSVMSGD(data,
      numFeatures,
      numIterations,
      stepSize,
      regParam)
  }
}

object BilinearSVM {

  private def trainBilinearSVMLBFGS(data: RDD[(Double, Vector)],
                                    numFeatures: Int,
                                    numIterations: Int,
                                    regParam: Double
                           ): BilinearSVMModel = {
    val gradient = new BilinearSVMGradient(numFeatures)

    val initialWeights = Vectors.zeros(numFeatures * numFeatures + 1)

    val (modelWeights, losses) = LBFGS.runLBFGS(data = data,
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

  private def trainBilinearSVMSGD(data: RDD[(Double, Vector)],
                                  numFeatures: Int,
                                  numIterations: Int,
                                  stepSize: Double,
                                  regParam: Double
                      ): BilinearSVMModel = {
    val gradient = new BilinearSVMGradient(numFeatures)

    val initialWeights = Vectors.zeros(numFeatures * numFeatures + 1)

    val (modelWeights, losses) = GradientDescent.runMiniBatchSGD(data,
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