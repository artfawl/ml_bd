package LinReg

import breeze.linalg._
import breeze.stats._
import breeze.stats.distributions.Gaussian
import java.io._



class LinearRegression() {
  private var _weights: DenseMatrix[Double] = new DenseMatrix[Double](0, 0)
  private var _bias: Double = 0

  def fit(X: DenseMatrix[Double], y: DenseMatrix[Double], batchSize: Int, epochs: Int=100, lr: Double=1e-3) {
    println("start fitting")
    assert(X.rows==y.rows)
    assert(batchSize<=X.rows)
    _weights = DenseMatrix.rand(rows = X.cols, cols=1, rand=distributions.Gaussian(0,1))

    for (epoch <- 1 to epochs) {
      for (i <- 0 to X.rows / batchSize) {
        val first = i * batchSize
        val last = if ((i+1)*batchSize <= X.rows) (i+1) * batchSize-1 else X.rows-1
        val xBatch = X(first to last, ::)
        val yBatch = y(first to last, ::)
        val wGrad = xBatch.t * (xBatch * _weights + _bias - yBatch)
        val bGrad = (xBatch * _weights +_bias - yBatch).toDenseVector.reduce((x,y)=>x+y)
        _weights -= lr * wGrad
        _bias -= lr * bGrad
      }
    }
  }

  def predict(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    return X * _weights + _bias
  }

  def bias: Double = _bias
  def weights: DenseMatrix[Double] = _weights
}
