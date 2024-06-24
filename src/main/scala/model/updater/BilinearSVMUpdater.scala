package model.updater

import dev.ludovic.netlib.blas.BLAS
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.Updater

class BilinearSVMUpdater extends Updater {
  private lazy val blas = BLAS.getInstance()

  override def compute(weightsOld: Vector, gradient: Vector, stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)

    val weightsCopy = weightsOld.copy
    blas.daxpy(weightsCopy.size, -thisIterStepSize, gradient.toArray, 1, weightsCopy.toArray, 1)
    blas.daxpy(weightsCopy.size - 1, -thisIterStepSize * regParam, weightsCopy.toArray, 1, weightsCopy.toArray, 1)
    val norm = blas.dnrm2(weightsCopy.size - 1, weightsCopy.toArray, 1)

    val regValue = 0.5 * regParam * norm * norm

    (weightsCopy, regValue)
  }
}

