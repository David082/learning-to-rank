package spark.preprocessing

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SaveMode, SparkSession}

import scala.collection.mutable

/**
  * Created by yu_wei on 2018/2/6.
  */
object FeatureImportance {

   def importance(sc: SparkContext, features: List[String], hdfsPath: String) = {
      val xgb = XGBoost.loadModelFromHadoopFile(hdfsPath)(sc)
      val featuresMap = xgb.booster.getFeatureScore()
      val featureNameMap = new mutable.HashMap[String, Int]()
      for (i <- 1 to features.length) {
         var key = "f" + (i - 1)
         if (featuresMap.keySet.contains(key))
            featureNameMap += features.apply(i - 1) -> featuresMap.get(key).get.toInt
         else
            featureNameMap += features.apply(i - 1) -> 0
      }
      featureNameMap
   }

   /**
     * save feature importances to hive table
     *
     * @param modelPath : model HDFS path
     * @param features  : feature list
     * @param tablename : dst table name
     */
   def saveImportanceAsHive(spark: SparkSession, modelPath: String, features: List[String], tablename: String) = {
      import spark.implicits._

      val xgb = XGBoost.loadModelFromHadoopFile(modelPath)(spark.sparkContext)
      val featuresMap = xgb.booster.getFeatureScore()
      val fmap = new mutable.HashMap[String, Int]()
      for (i <- 1 to features.length) {
         var key = "f" + (i - 1)
         if (featuresMap.keySet.contains(key))
            fmap += features.apply(i - 1) -> featuresMap.get(key).get.toInt
         else
            fmap += features.apply(i - 1) -> 0
      }

      fmap.foreach(println)
      val featureImportancesDF = fmap.toSeq.toDF("feature", "Importance")
      // dw_htlbizdb.tmp_yw_featureImportances
      featureImportancesDF.repartition(6).write.mode(SaveMode.Overwrite).format("orc").saveAsTable(tablename)
   }

}
