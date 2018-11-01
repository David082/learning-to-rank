package hotelrank

import org.apache.spark.sql._
import hotelrank.FeatureList._
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import spark.preprocessing.FeatureTransform._
import spark.metrics.RankMetrics._

/**
  * Created by yu_wei on 2018/11/1.
  */
object XGBModel {

   def main(args: Array[String]): Unit = {

      val spark = SparkSession.builder()
        .appName("spark xgboost model training")
        .master("yarn")
        .enableHiveSupport()
        .getOrCreate()

      // ------ param
      val at = 3
      val label = "booking_bool"
      val keyCol = "qid_mainkey"
      val features = FEATURES
      val table = "dw_htlbizdb.htllist_user_query_yxjd_dataset_basic"

      // ------ fitting
      var trainSet = getPoiTrainSet(spark, table, features)

      val params = List(
         "booster" -> "gbtree",
         "tree_method" -> "exact",
         //"rate_drop" -> 0.1,
         //"skip_drop" -> 0.5,
         "objective" -> "binary:logistic",
         "grow_policy" -> "lossguide",
         //"tree_method" -> "hist", // xgboost4j-spark does not support fast histogram for now
         //"scale_pos_weight" -> scalePosWeight,
         //"weightCol" -> "sample_weight",
         "eval_metric" -> "auc",
         "max_depth" -> 6,
         "eta" -> 0.25,
         "subsample" -> 0.8,
         "colsample_bytree" -> 0.8,
         "min_child_weight" -> 1,
         "alpha" -> 0.05,
         "gamma" -> 3,
         "silent" -> 0
      ).toMap
      val xgbModel = XGBoost.trainWithDataFrame(trainSet, params,
         round = 200, nWorkers = 200,
         featureCol = "features", labelCol = label)
      xgbModel.saveModelAsHadoopFile("hdfs:///path/xgb.model")(spark.sparkContext)

      // ------ model eval
      val testSet = getPoiTestSet(spark, table, features)
      var predict: DataFrame = xgbModel.transform(testSet) // predict
      predict = predict.withColumn("probs", colprob(col("probabilities")).cast(DoubleType))

      val auc = AUC(predict, "probs", labelCol = label)
      val ndcg = ndcgAtK(predict, keyCol, k = 3, label, "new_score")
      ndcg.show(false)
      val precision = precisionAtK(predict, keyCol, k = 3, label, "new_score")
      precision.show(false)

      spark.stop()

   }

   def getPoiTrainSet(spark: SparkSession, table: String, features: List[String]): DataFrame = {
      val sql = s"select * from $table" +
        " where d >= '2018-10-11' and d <= '2018-10-17'" +
        " and isorder = 1 and sort_qry = 9 and scene_foreign = 1"

      var df = spark.sql(sql)
      df = df.withColumn("dist", colcoder(col("dist")))
      df = featureAssembler(df, features)
      df
   }

   def getPoiTestSet(spark: SparkSession, table: String, features: List[String]): DataFrame = {
      val sql = s"select * from $table" +
        " where d >= '2018-10-19' and d <= '2018-10-20'" +
        " and isorder = 1 and sort_qry = 9 and scene_foreign = 1"

      var df = spark.sql(sql)
      df = df.withColumn("dist", colcoder(col("dist")))
      df = featureAssembler(df, features)
      df
   }

}
