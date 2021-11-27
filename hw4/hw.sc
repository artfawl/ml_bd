import org.apache.spark.sql._
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .master("local[*]")
  .appName("hw4")
  .getOrCreate()

import spark.implicits._

val df = spark.read.csv("/home/anaxagoras/MADE/ml_bd/ml_bd/hw4/data.csv").withColumn("id",monotonically_increasing_id())
val sz = df.count()
var preprocessed = df.withColumn("lower",
  regexp_replace(lower(col("_c0")), "[^a-zA-Z0-9\\s.-]", ""))

preprocessed = preprocessed.withColumn("splitted", split(col("lower"), " "))
preprocessed = preprocessed.select(col("id"),col("splitted"), explode(col("splitted")))

val grouped = preprocessed.groupBy($"id", $"splitted", $"col").count()
val tf = grouped.withColumn("tf", col("count")/size($"splitted"))

val gr_by_words = preprocessed.dropDuplicates().groupBy($"col").count()
val with_idf = gr_by_words.withColumn("idf", log(pow(col("count"), -1)*sz))
val top_idf = with_idf.sort(desc("count")).limit(100)

val fin = tf.join(top_idf, "col")
  .withColumn("tf-idf", col("tf")*col("idf")).select("id","tf-idf","col")


val res = fin.groupBy("id").pivot("col").sum("tf-idf")
res.show()