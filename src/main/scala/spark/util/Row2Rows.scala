package spark.util

import org.apache.spark.sql.{Row, SparkSession, _}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by yu_wei on 2018/10/30.
  *
  * -- column size
  * https://stackoverflow.com/questions/44541605/how-to-get-the-lists-length-in-one-column-in-dataframe-spark?rq=1
  * -- row to rows
  * https://www.jianshu.com/p/b2346432abc1
  * -- ArrayType
  * https://medium.com/@mrpowers/working-with-spark-arraytype-and-maptype-columns-4d85f3c8b2b3
  *
  */
object Row2Rows {

   def splitByResponse(version: String, seqid: String, request: String, response: String, fsize: Int) = {
      val rqends = request.length
      val rpends = response.length
      val requestList = request.substring(1, rqends - 1).split(",")
      val responseList = response.substring(1, rpends - 1).split(",")

      val rsize = requestList.size / fsize
      var resSeq = Seq[Row]()
      for (r <- 0 until rsize) {
         var row = new ArrayBuffer[String]()
         for (i <- 0 until fsize) {
            row += requestList.apply(r * fsize + i)
         }
         resSeq = resSeq :+ Row(version, seqid, row, responseList.apply(r).toDouble)
      }
      resSeq
   }

   def versionTable(spark: SparkSession, df: DataFrame, version: String): DataFrame = {
      var data = df.select("version", "seqid", "status", "request", "response")
      data = data.withColumn("features", split(col("status"), "\\,"))
        .withColumn("f_length", size(col("features")))
        .where(s"version = '$version'")
      val fize = data.select("f_length").first().getInt(0)

      val resRdd = data.select("version", "seqid", "request", "response").rdd.flatMap(line => {
         var v = line.getAs[String]("version")
         var id = line.getAs[String]("seqid")
         var rq = line.getAs[String]("request")
         var rp = line.getAs[String]("response")
         splitByResponse(v, id, rq, rp, fize)
      })

      val schema = StructType(List(
         StructField("version", StringType, nullable = false),
         StructField("seqid", StringType, nullable = true),
         StructField("request", ArrayType(StringType, true), nullable = true),
         StructField("response", DoubleType, nullable = true)
      ))
      spark.createDataFrame(resRdd, schema)
   }
}
