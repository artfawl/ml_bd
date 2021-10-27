package LinReg

import LinReg._
import breeze.linalg._
import java.io._

object Main {
  def read_dataset(path: String): Array[DenseMatrix[Double]]= {
    val train = csvread(
      new File(
        path
      ),
      separator = ',', skipLines = 1)(::, 1 to -1)
    val X = train(::, 0 to -2)
    val y: DenseMatrix[Double] = train(::, -1).asDenseMatrix.t
    return Array(X, y)
  }
  def main(args: Array[String]): Unit = {
    val trainData: Array[DenseMatrix[Double]] = read_dataset(args(0))
    val xTrain = trainData(0)
    val yTrain = trainData(1)
    val model = new LinearRegression()
    model.fit(xTrain, yTrain, 64)
    val testData: Array[DenseMatrix[Double]] = read_dataset(args(1))
    val xTest = testData(0)
    val yTest = testData(1)
    val yPred = model.predict(xTest)
    val er = yTest - yPred
    print("error: ")
    val tmp = er.t*er
    println(scala.math.pow((tmp(0,0)/yTest.rows.toDouble), 0.5))
    csvwrite(new File(args(2)), yPred, separator = ',')
  }

}
