package model

import model.gradient.BilinearSVMGradient
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}

class BilinearSVMModel(val weightMatrix: DenseMatrix,
                       val intercept: Double
                      ) extends Serializable {

  private val splitIndex = weightMatrix.numRows

  private def predictPoint(dataVector: Vector, weightMatrix: DenseMatrix, intercept: Double, splitIndex: Int): Double = {
    if (dataVector.size == weightMatrix.numRows) {
      BilinearSVMGradient.xTAx(dataVector.toArray, weightMatrix.toArray) + intercept
    } else {
      val x1 = dataVector.toArray.slice(0, splitIndex)
      val x2 = dataVector.toArray.slice(splitIndex, dataVector.size)
      BilinearSVMGradient.x1TAx2(x1, weightMatrix.toArray, x2) + intercept
    }
  }

  def predict(data: RDD[Vector]): RDD[Double] = {

    val localWeights = weightMatrix
    val bcWeights = data.context.broadcast(localWeights)
    val localIntercept = intercept
    val localSplitIndex = splitIndex
    data.mapPartitions { iter =>
      val w = bcWeights.value
      iter.map(v => predictPoint(v, w, localIntercept, localSplitIndex))
    }
  }

  def predict(data: Vector): Double = {
    predictPoint(data, weightMatrix, intercept, splitIndex)
  }

  def save(path: String): Unit = {
    val sc: SparkContext = SparkContext.getOrCreate()
    sc.parallelize(Seq(ModelData(weightMatrix = weightMatrix, intercept = intercept)), 1)
      .saveAsObjectFile(path)

  }
}

object BilinearSVMModel {
  def load(path: String): BilinearSVMModel = {
    val sc: SparkContext = SparkContext.getOrCreate()
    val modelData = sc.objectFile[ModelData](path)
      .first()

    new BilinearSVMModel(modelData.weightMatrix, modelData.intercept)
  }
}

private case class ModelData(weightMatrix: DenseMatrix, intercept: Double)