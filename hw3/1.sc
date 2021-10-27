import breeze.linalg._
import breeze.stats._
import breeze.stats.distributions.Gaussian

import java.io._
import scala.io.Source._

val train = csvread(
  new File(
    "/home/anaxagoras/MADE/ml_bd/hw3/files/train.csv"
  ),
  separator = ',', skipLines = 1)(::, 1 to -1)

//train(::, 0 to -2)
val X = DenseMatrix.horzcat(DenseMatrix.ones(train.rows, 1): DenseMatrix[Double], train(::, 0 to -2))
val y: DenseMatrix[Double] = train(::, -1).asDenseMatrix.t

val norm = distributions.Gaussian(0,1)
var w = DenseMatrix.rand(rows = X.cols, cols=1, rand=norm)
val Epochs = 100
val BatchSize = 64
val lr = 1e-3
for (epoch <- 1 to Epochs) {
  for (i <- 0 to X.rows / BatchSize) {
    val first = i * BatchSize
    val last = if ((i+1)*BatchSize <= X.rows) (i+1) * BatchSize-1 else X.rows-1
    //print(first, last)
    val batch_x = X(first to last, ::)
    //print("lol")
    val batch_y = y(first to last, ::)
    //println("kek")
    //println(batch_x.cols, w.rows)
    val grad = batch_x.t * batch_x * w - batch_x.t * batch_y
    //println("kek")
    w -= lr * grad
  }
}

val tmp = X * w - y
tmp.t * tmp / X.rows.toDouble
w