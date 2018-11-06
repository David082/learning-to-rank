package spark.util

/**
  * Created by yu_wei on 2018/11/6.
  */
object ParamArgs {

   def main(args: Array[String]): Unit = {

      // ------ Args
      var label = ""
      var table = ""
      var round = 0

      args.sliding(2, 2).toList.collect {
         case Array("--label", argLabel: String) => label = argLabel
         case Array("--table", argTable: String) => table = argTable
         case Array("--round", argRound: String) => round = argRound.toInt
      }

   }

}
