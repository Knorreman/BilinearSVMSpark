package model.gradient

import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.Gradient

class BilinearSVMGradient(val numFeatures: Int) extends Gradient {
  private lazy val blas: BLAS = BLAS.getInstance()

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    if (data.size == numFeatures) {
      val x = data.toArray
      val xLen = x.length
      val aArray = weights.toArray.slice(0, xLen * xLen)
      val b = weights(weights.size - 1)
      val prediction = BilinearSVMGradient.xTAx(x, aArray) + b

      if (label * prediction < 1) {
        val gradientA = aArray.clone()
        blas.dger(xLen, xLen, -label, x, 1, x, 1, gradientA, numFeatures)
        val gradient = Vectors.dense(gradientA ++ Array(-label))

        (gradient, 1 - label * prediction)
      } else {
        (Vectors.zeros(weights.size), 0.0)
      }
    } else {
        val splitIndex = numFeatures
        val aArray = weights.toArray.slice(0, numFeatures * numFeatures)
        val x1 = data.toArray.slice(0, splitIndex)
        val x2 = data.toArray.slice(splitIndex, data.size)

        val b = weights(weights.size - 1)
        val prediction = BilinearSVMGradient.x1TAx2(x1, aArray, x2) + b

        if (label * prediction < 1) {
          val gradientA = aArray.clone()
          blas.dger(numFeatures, numFeatures, -label, x1, 1, x2, 1, gradientA, numFeatures)
          val gradient = Vectors.dense(gradientA ++ Array(-label))

          (gradient, 1 - label * prediction)
        } else {
          (Vectors.zeros(weights.size), 0.0)
        }
    }

  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    val (gradient, loss) = compute(data, label, weights)
    blas.daxpy(gradient.size, 1.0, gradient.toArray, 1, cumGradient.toArray, 1)
    loss
  }


}

object BilinearSVMGradient {
  private lazy val blas: BLAS = BLAS.getInstance()
  private[model] def xTAx(x: Array[Double], A: Array[Double]): Double = {
    val numFeatures = x.length
    val y = new Array[Double](numFeatures)

    blas.dgemv("N", numFeatures, numFeatures, 1.0, A, numFeatures, x, 1, 0.0, y, 1)

    val result = blas.ddot(numFeatures, x, 1, y, 1)
    result
  }

  private[model] def x1TAx2(x1: Array[Double], A: Array[Double], x2: Array[Double]): Double = {
    val numFeatures = x1.length
    val y = new Array[Double](numFeatures)

    blas.dgemv("N", numFeatures, numFeatures, 1.0, A, numFeatures, x2.toArray, 1, 0.0, y, 1)

    val result = blas.ddot(numFeatures, x1.toArray, 1, y, 1)
    result
  }
}
