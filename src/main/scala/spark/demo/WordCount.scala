package spark.demo

import org.apache.spark.sql.SparkSession

/**
  * Created by yu_wei on 2018/07/26.
  */
object WordCount {

   def main(args: Array[String]): Unit = {

      val spark = SparkSession.builder()
        .master("local")
        .appName("word count")
        .getOrCreate()

      val lines = spark.sparkContext.textFile("hdfs://spark.txt")
      val words = lines.flatMap(line => line.split(" "))
      val pairs = words.map(word => (word, 1))
      val wordCounts = pairs.reduceByKey(_ + _) // shuffle

      wordCounts.foreach(wordCount => println(wordCount._1 + "appears " + wordCount._2 + "times."))

      spark.stop()

   }

}
