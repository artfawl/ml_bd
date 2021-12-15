import breeze.linalg._
import breeze.stats.distributions
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg

import java.util.Random
//import org.apache.spark.ml.

val spark = SparkSession.builder()
  .master("local[*]")
  .appName("hw5")
  .getOrCreate()

trait LRParams extends Params {
  final val inputCol= new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val labelCol = new Param[String](this, name="labelCol", "Label Column")
  final val lr = new Param[Double](this, name="lr", "Learning rate")
  final val iters = new Param[Integer](this, name="iters", "Number of iterations")
}

class LRModel(
               override val uid: String, w: DenseMatrix[Double], b:  Double) extends Model[LRModel] with LRParams {

  def setInputCol(value: String) = set(inputCol, value)

  def setOutputCol(value: String) = set(outputCol, value)

  def setLabelCol(value: String) = set(labelCol, value)

  def setLR(value: Double) = set(lr, value)

  def setIters(value: Integer) = set(iters, value)

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    schema.add(StructField($(outputCol), DoubleType, false))
  }
  override def copy(extra: ParamMap): LRModel = {
    defaultCopy(extra)
  }
  override def transform(dataset: Dataset[_]): DataFrame = {
    val temp = dataset.select(col($(inputCol)))
      .collect()
      .map(_(0))
      .map(_.asInstanceOf[linalg.DenseVector].toArray)
      .flatten

    val c: Integer = dataset.count().toInt
    val r: Integer = temp.length / c

    val X: DenseMatrix[Double] = new DenseMatrix(r, c, temp).t

    val yPred = (X * w + b).toDenseVector.toArray.toSeq.map(el=>Row(el))

    val Schema = new StructType().add($(outputCol),DoubleType)

    val results = spark.createDataFrame(
      spark.sparkContext.parallelize(yPred),
      Schema
    ).withColumn("tmp",monotonically_increasing_id())

    val tmp = dataset.withColumn("tmp",monotonically_increasing_id())

    return tmp.join(results, "tmp")
  }
}

class LR(val uid: String) extends Estimator[LRModel] with LRParams {


  def setInputCol(value: String) = set(inputCol, value)

  def setOutputCol(value: String) = set(outputCol, value)

  def setLabelCol(value: String) = set(labelCol, value)

  def setLR(value: Double) = set(lr, value)

  def setIters(value: Integer) = set(iters, value)

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(ds: Dataset[_]): LRModel = {
    val temp = ds.select(col($(inputCol)))
      .collect()
      .map(_(0))
      .map(_.asInstanceOf[linalg.DenseVector].toArray)
      .flatten

    val temp2 = ds.select(col($(labelCol)))
      .collect()
      .map(_(0).asInstanceOf[Double])

    val c: Integer = ds.count().toInt
    val r: Integer = temp.length / c

    val X: DenseMatrix[Double] = new DenseMatrix(r, c, temp).t
    val y: DenseMatrix[Double] = new DenseMatrix(1, c, temp2).t

    var w: DenseMatrix[Double] = DenseMatrix.rand(r, 1, rand=distributions.Gaussian(0,1))
    var b: Double = 0

    for(iter <- 1 to $(iters)) {
      val wGrad = X.t * (X * w + b - y)
      val bGrad = (X * w + b - y).toDenseVector.reduce((x,y)=>x+y)

      w -= $(lr) * wGrad
      b -= $(lr) * bGrad(0)
    }



  var model = new LRModel(uid, w, b)
    .setOutputCol($(outputCol))
    .setInputCol($(inputCol))
    .setLR($(lr))
    .setIters($(iters))
    .setLabelCol($(labelCol))

  return model
  }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    schema.add(StructField($(outputCol), DoubleType, false))
  }

  override def copy(extra: ParamMap) = {
    defaultCopy(extra)
  }

}

val df = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/home/anaxagoras/MADE/ml_bd/ml_bd/hw5/clean_train.csv")
  .withColumnRenamed("5", "label")


val assembler = new VectorAssembler()
  .setInputCols(Array("0","1","2","3","4"))
  .setOutputCol("features")



val output = assembler.transform(df)

val est = new LR()
  .setInputCol("features")
  .setLR(0.001)
  .setIters(1000)
  .setOutputCol("res")
  .setLabelCol("label")

val trans = est.fit(output)

val rs = trans.transform(output)

rs.show()

// В колонке label такие же значения как и в res
// Алгоритм работает
