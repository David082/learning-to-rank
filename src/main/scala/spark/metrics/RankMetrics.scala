package spark.metrics

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

/**
  * Created by yu_wei on 2018/7/18.
  */
object RankMetrics {

   /**
     * AUC
     *
     * @param predictedDF : DataFrame after predicted
     * @param scoreCol    : predict column
     * @param labelCol    : label column
     * @return AUC
     */
   def AUC(predictedDF: DataFrame, scoreCol: String, labelCol: String) = {
      val scoreAndLabels: RDD[(Double, Double)] = predictedDF.select(scoreCol, labelCol).rdd.map {
         case Row(score: Double, label: Int) => (score.toDouble, label.toDouble)
      }
      val metric = new BinaryClassificationMetrics(scoreAndLabels)
      val auc = metric.areaUnderROC()
      auc
   }

   def reli = udf((i: Int) => {
      Math.pow(2, i) - 1
   })

   def log2 = udf((x: Int) => {
      Math.log(x + 1) / Math.log(2)
   })

   def relDivLog2AtK = udf((rank: Int, k: Int, rel: Int, logx: Double) => {
      if (rank > k) 0
      else if (logx == 0 || rel.toString == null || logx.toString == null) 0
      else rel / logx
   })

   def qidNdcgAtK(df: DataFrame, keyofRank: String, k: Int, label: String, predictions: String = null): DataFrame = {
      if (predictions == null) {
         val dcgAtK = df.select(keyofRank, "rank_in_qid", label)
           .withColumn("dcgAtK", relDivLog2AtK(col("rank_in_qid"), lit(k), reli(col(label)), log2(col("rank_in_qid"))))
           .withColumn("ideal_rank", row_number().over(Window.partitionBy(keyofRank).orderBy(col(label).desc)))
           .withColumn("idcgAtK", relDivLog2AtK(col("ideal_rank"), lit(k), reli(col(label)), log2(col("ideal_rank"))))
           .groupBy(keyofRank).agg((sum("dcgAtK") / sum("idcgAtK")).alias("ndcg"))
         dcgAtK
      } else {
         val dcgAtK = df.select(keyofRank, label, predictions)
           .withColumn("dcg_rank", row_number().over(Window.partitionBy(keyofRank).orderBy(col(predictions).desc)))
           .withColumn("ideal_rank", row_number().over(Window.partitionBy(keyofRank).orderBy(col(label).desc)))
           .withColumn("dcgAtK", relDivLog2AtK(col("dcg_rank"), lit(k), reli(col(label)), log2(col("dcg_rank"))))
           .withColumn("idcgAtK", relDivLog2AtK(col("ideal_rank"), lit(k), reli(col(label)), log2(col("ideal_rank"))))
           .groupBy(keyofRank).agg((sum("dcgAtK") / sum("idcgAtK")).alias("ndcg"))
         dcgAtK
      }
   }

   def ndcgAtK(df: DataFrame, keyofRank: String, k: Int, label: String, predictions: String = null) = {
      qidNdcgAtK(df, keyofRank, k, label, predictions).agg(avg(col("ndcg")).alias("ndcg"))
   }

   def relPrecision = udf((rank: Int, k: Int, rel: Int) => {
      if (rank > k) 0
      else rel
   })

   def qidPrecisionAtK(df: DataFrame, keyofRank: String, k: Int, label: String, predictions: String = null) = {
      if (predictions == null) {
         val preAtK = df.select(keyofRank, "rank_in_qid", label)
           .withColumn("positiveAtK", relPrecision(col("rank_in_qid"), lit(k), col(label)))
           .groupBy(keyofRank).agg((sum("positiveAtK") / sum(label)).alias("precisionAtK"))
         preAtK
      } else {
         val preAtK = df.select(keyofRank, label, predictions)
           .withColumn("rank_in_pred", row_number().over(Window.partitionBy(keyofRank).orderBy(col(predictions).desc)))
           .withColumn("positiveAtK", relPrecision(col("rank_in_pred"), lit(k), col(label)))
           .groupBy(keyofRank).agg((sum("positiveAtK") / sum(label)).alias("precisionAtK"))
         preAtK
      }

   }

   def precisionAtK(df: DataFrame, keyofRank: String, k: Int, label: String, predictions: String = null) = {
      qidPrecisionAtK(df, keyofRank, k, label, predictions).agg(avg(col("precisionAtK")).alias("precisionAtK"))
   }

}
