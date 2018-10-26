package predictor;

import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.DMatrix;
import ml.dmlc.xgboost4j.scala.spark.XGBoost;
import ml.dmlc.xgboost4j.scala.spark.XGBoostModel;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/*
 * Created by yu_wei on 2018/5/23.
 *
 * https://www.programcreek.com/java-api-examples/?api=ml.dmlc.xgboost4j.java.DMatrix
 *
 * -- args case:
 * https://www.cnblogs.com/sinceForever/p/8454385.html
 *
 */
public class XGBPredict {

    // args[0]: MODEL_PATH -- model path
    // args[1]: CSV_PATH -- file path of predict data
    // args[2]: filePath -- file path of result data

    private static final String MODEL_PATH = "file:///home/hotel/yw/xgboostpredictor/poi_model_20180521";
    private static final String CSV_PATH = "/home/hotel/yw/xgboostpredictor/sample.csv";
    private static final String SAVE_PATH = "/home/hotel/yw/xgboostpredictor/res.csv";

    public static void main(String[] args) throws IOException, XGBoostError {

        // Spark conf
        SparkConf conf = new SparkConf().setAppName("local xgb predict").setMaster("local");
        SparkContext sc = new SparkContext(conf);
        XGBoostModel xgb = XGBoost.loadModelFromHadoopFile(args[0], sc);
        Booster booster = xgb.booster();

        DataBean bean = dMatrixFromCsv(args[1], Float.NaN); /* Specify the missing value is necessary */
        float[][] predict = booster.predict(bean.dataMat, false, 0);

        // Predict and then save result
        savePredictToCsv(args[2], bean, predict);

    }

    /**
     * Read data from csv file and then change it into DMatrix.
     *
     * @param filePath: The file path of predict data set
     * @param missing:  Setting the missing value
     */
    private static DataBean dMatrixFromCsv(String filePath, float missing) throws IOException, XGBoostError {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        Map<Integer, float[]> sample = new HashMap<>();
        String[] nameArr = br.readLine().split(",");

        // Read csv data as map
        String paramLine = null;
        int numrow = 0;
        while ((paramLine = br.readLine()) != null) {
            String[] paramLineArr = paramLine.split(",");
            float[] row = new float[paramLineArr.length];
            for (int i = 0; i < paramLineArr.length; i++) {
                row[i] = Float.parseFloat(paramLineArr[i]);
            }
            sample.put(numrow, row);
            numrow += 1;
        }

        float[] data = new float[numrow * (nameArr.length - 1)];
        for (int r = 0; r < sample.size(); r++) {
            float[] row = sample.get(r);
            System.arraycopy(row, 0, data, r * (nameArr.length - 1), (nameArr.length - 1));
        }

        DataBean bean = new DataBean();
        bean.dataMat = new DMatrix(data, numrow, nameArr.length - 1, missing);

        bean.colnames = nameArr;
        bean.sample = sample;

        return bean;
    }

    /**
     * Setting the missing value, 'false' represent Float.Nan, else input to float.
     * @param missingInput: false -- Float.Nan; else input missing.
     */
    private static float setMissing(String missingInput) {
        float missing = 0;
        switch (missingInput.toLowerCase()) {
            case "false":
                missing = Float.NaN;
                break;
            default:
                missing = Float.parseFloat(missingInput);
        }
        System.out.println("====== missing value is: " + missing);
        return missing;
    }

    /**
     * Saving the predict result.
     *
     * @param filePath: Save result to this file
     * @param bean:     predict data set
     * @param predict:  predict scores
     */
    private static void savePredictToCsv(String filePath, DataBean bean, float[][] predict) throws IOException {
        File file = new File(filePath);
        if (file.isFile() && file.exists()) {
            file.delete();
        }
        FileWriter fw = new FileWriter(filePath, true);
        BufferedWriter bw = new BufferedWriter(fw);

        // Write col names
        for (int c = 0; c < bean.colnames.length; c++) {
            bw.write(bean.colnames[c] + ", ");
        }
        bw.write("predict");
        bw.newLine();

        // Write data
        for (int i = 0; i < predict.length; i++) {
            for (int col = 0; col < bean.colnames.length; col++) {
                bw.write(bean.sample.get(i)[col] + ", ");
            }
            bw.write(String.valueOf(predict[i][0]));
            bw.newLine();
            System.out.println("[predict] " + predict[i][0] + " -- [online] " + bean.sample.get(i)[bean.colnames.length - 1]);
        }
        bw.close();
        fw.close();
    }

}
