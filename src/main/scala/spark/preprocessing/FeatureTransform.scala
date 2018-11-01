package spark.preprocessing

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
  * Created by yu_wei on 2018/11/1.
  */
object FeatureTransform {

   def colcoder = udf((arg: String) => {
      if (arg == "<100") 100.0 else arg.toDouble
   })

   /**
     * return featureCol and labelCol, and then for train or predict
     *
     * @param df       :data set
     * @param features : features
     * @return
     */
   def featureAssembler(df: DataFrame, features: List[String]): DataFrame = {
      val data = df.na.fill(-999999.0)
      val vectorAssembler = new VectorAssembler()
        .setInputCols(features.toArray)
        .setOutputCol("features")
      val pipeline = new Pipeline().setStages(Array(vectorAssembler))
      val dataAssembler = pipeline.fit(data).transform(data)
      dataAssembler
   }

   def colprob = udf((probs: Vector) => {
      probs(1)
      // probs.toDense.apply(1).formatted("%.6f")
   })

}
