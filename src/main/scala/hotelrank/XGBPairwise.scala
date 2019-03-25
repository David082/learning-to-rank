package hotelrank

import org.apache.spark.sql._
import hotelrank.FeatureList._
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField}
import spark.preprocessing.FeatureTransform._
import spark.metrics.RankMetrics._

import scala.collection.mutable.ListBuffer

/**
  * Created by yu_wei on 2019/2/22.
  *
  * 参数说明
  * [1] weightCol: 设置样本权重
  *
  * [2] pairwise模型
  * 第一步: 对训练集按照qid进行重排
  * 第二步: 计算各个qid的长度
  * 第三步:
  * 指定 objective -> rank:pairwise
  * 指定 eval_metric -> ndcg
  *
  * [3] missing
  * 预测时调用predict
  *
  */
object XGBPairwise {

   val LABEL = "booking_bool"

   def main(args: Array[String]): Unit = {

      val spark = SparkSession.builder()
        .appName("spark xgboost model training")
        .master("yarn")
        .enableHiveSupport()
        .getOrCreate()

      val features = FEATURES

      var trainSet = getPoiTrainSet(spark, "di_dsjobdb.table", features)
      trainSet = trainSet.withColumn("sample_weight", sampleWeight(col(""), col(""))) // sample weight

      // ------ get group id :
      val queryCount = trainSet.groupBy("qid_mainkey").count().select("qid_mainkey", "count")
        .withColumnRenamed("qid_mainkey", "qid_mainkey_new")
      val trainWithGroupID = trainSet.join(queryCount, col("qid_mainkey") === col("qid_mainkey_new"), "left_outer")

      // ------ set group id :
      val groupQueryID = calcQueryGroups(trainWithGroupID, "qid_mainkey") // Seq(Seq(Int))

      val params = List(
         "booster" -> "gbtree",
         "tree_method" -> "exact",
         //"rate_drop" -> 0.1,
         //"skip_drop" -> 0.5,
         "objective" -> "rank:pairwise", // rank:pairwise
         "grow_policy" -> "lossguide",
         //"scale_pos_weight" -> scalePosWeight,
         "groupData" -> groupQueryID,  // 设置qid group, 对应python version xgboost set_group
         "weightCol" -> "sample_weight", // 设置样本权重
         "eval_metric" -> "ndcg", // ndcg -> 对应pairwise模型
         "max_depth" -> 6,
         "eta" -> 0.25,
         "subsample" -> 0.8,
         "colsample_bytree" -> 0.8,
         "min_child_weight" -> 1,
         "alpha" -> 0.05,
         "gamma" -> 3,
         "silent" -> 0
      ).toMap
      val xgbModel = XGBoost.trainWithDataFrame(trainWithGroupID, params,
         round = 200, nWorkers = 200,
         missing = -999999.0f,
         featureCol = "features", labelCol = LABEL)

      val testSet = getPoiTrainSet(spark, "di_dsjobdb.table", features)

      // ------ predict with missing
      val testingData = testSet.select(LABEL, features: _*).rdd.map { row =>
         val rowToArray = row.toSeq.toArray.map(_.toString.toDouble)
         // val label = rowToArray(0)
         val featuresArray = rowToArray.drop(1)
         val features = Vectors.dense(featuresArray).toDense
         features
      }
      val predictResults = xgbModel.predict(testingData, -999999.0f).zip(testSet.rdd).map {
         case (predictColumn: Array[Float], originalColumns: Row) =>
            Row.fromSeq(originalColumns.toSeq :+ predictColumn.apply(0).toDouble)
      }
      val schema = testSet.schema.add(StructField("probability", DoubleType))
      var probs = spark.createDataFrame(predictResults, schema)


      spark.stop()
   }

   /**
     * 计算训练集对应的qid group
     * @param ds : 训练集
     * @param queryIdCol : qid
     * @return : 返回训练集中qid对应的长度，格式为list
     */
   def calcQueryGroups(ds: DataFrame, queryIdCol: String): Seq[Seq[Int]] = {
      val groupData = ds.select(col(queryIdCol)).rdd.mapPartitionsWithIndex {
         (partitionId, rows) =>
            rows.foldLeft(List[(String, Int)]()) {
               (acc, e) =>
                  val queryId = e.getAs[String](queryIdCol)
                  acc match {
                     // If the head matches current queryId increment it
                     case ((`queryId`, count) :: xs) => (queryId, count + 1) :: xs
                     // otherwise add new item to head
                     case _ => (queryId, 1) :: acc
                  }
               // Replace query id with partition id
            }.map {
               case (_, count) => (partitionId, count)
               // reverse because the list was built backwards
            }.reverse.toIterator
      }.collect()

      val numPartitions = ds.rdd.getNumPartitions
      val groups = Array.fill(numPartitions)(ListBuffer[Int]())
      // Convert our list of (partitionid, count) pairs into the result
      // format. Spark guarantees to sort order has been maintained.
      for (e <- groupData) {
         groups(e._1) += e._2
      }
      groups
   }

   def sampleWeight = udf((rank_in_qid: Int, booking: Int, predict: Double) => {
      1.0
   })

   def getPoiTrainSet(spark: SparkSession, table: String, features: List[String]): DataFrame = {
      val sql = s"select * from $table" +
        " where d >= '2018-10-11' and d <= '2018-10-17'" +
        " and isorder = 1 and sort_qry = 9 and scene_foreign = 1"

      var df = spark.sql(sql)
      df = df.withColumn("dist", colcoder(col("dist")))
      df = featureAssembler(df, features)
      df
   }

}
