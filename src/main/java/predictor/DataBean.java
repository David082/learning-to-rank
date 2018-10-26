package predictor;

import ml.dmlc.xgboost4j.scala.DMatrix;

import java.util.Map;

/*
 * Created by yu_wei on 2018/5/23.
 */
class DataBean {
    String[] colnames;
    Map<Integer, float[]> sample;
    DMatrix dataMat;

    /*
    private static DataBean dMatrixFromCsv(String filePath, float missing) throws IOException, XGBoostError {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        Map<Integer, float[]> sample = new HashMap<>();
        String[] nameArr = br.readLine().split(",");

        // Read csv data as map
        String paramLine = null;
        int numrow = 0;
        while ((paramLine = br.readLine()) != null) {
            Map<String, Float> map = new HashMap<String, Float>();
            String[] paramLineArr = paramLine.split(",");
            float[] row = new float[paramLineArr.length];
            for (int i = 0; i < paramLineArr.length; i++) {
                map.put(nameArr[i], Float.parseFloat(paramLineArr[i]));
                row[i] = map.get(nameArr[i]);
            }
            sample.put(numrow, row);
            numrow += 1;
        }

        float[] data = new float[numrow * FEATURE_LIST.length];
        float[] scores = new float[numrow];
        for (int r = 0; r < sample.size(); r++) {
            float[] row = sample.get(r);
            System.arraycopy(row, 0, data, r * FEATURE_LIST.length, FEATURE_LIST.length);
            scores[r] = row[FEATURE_LIST.length];
        }

        DataBean bean = new DataBean();
        bean.dataMat = new DMatrix(data, numrow, FEATURE_LIST.length, missing);
        bean.scores = scores;

        return bean;
    }*/

    /*
    private static DMatrix labeledPointFromCsv(String filePath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        Map<Integer, float[]> sample = new HashMap<>();
        String[] nameArr = br.readLine().split(",");
        List<LabeledPoint> list = new ArrayList<LabeledPoint>();

        // Read csv data as map
        String paramLine = null;
        while ((paramLine = br.readLine()) != null) {
            String[] paramLineArr = paramLine.split(",");
            double[] row = new double[paramLineArr.length];
            for (int i = 0; i < paramLineArr.length; i++) {
                row[i] = Float.parseFloat(paramLineArr[i]);
            }
            Vector vec = Vectors.dense(row);
            float[] frow = new float[paramLineArr.length];
            for (int j = 0; j < paramLineArr.length; j++) {
                frow[j] = (float) vec.apply(j);
            }
            LabeledPoint lp = LabeledPoint.apply(0.0f, null, frow, 1.0f, -1, Float.NaN);
            list.add(lp);
        }
        scala.collection.Iterator<LabeledPoint> data = JavaConverters.asScalaIteratorConverter(list.iterator()).asScala();

        return new DMatrix(data, null);
    }*/
}
