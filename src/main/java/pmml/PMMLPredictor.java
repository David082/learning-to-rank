package pmml;

import org.dmg.pmml.FieldName;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class PMMLPredictor {

    public static void main(String[] args) throws IOException {

        String filePath = "testset.csv";
        Map<String, Map<FieldName, Float>> data = fieldFromCsv(filePath);
        /*
        for (Entry<String, Map<FieldName, Float>> d: data.entrySet()) {
            System.out.println(d);
        }*/

        String pmmlModelPath = "XGBClassifier.pmml";
        Map<String, Float> scores = predict(pmmlModelPath, data);
        for (Entry<String, Float> s: scores.entrySet()) {
            System.out.println(s);
        }

    }

    /**
     * Using pmml file to predict.
     * @param pmmlModelPath: pmml model path
     * @param data: predict data
     * @return : predict scores
     *
     */
    private static Map<String, Float> predict(String pmmlModelPath, Map<String, Map<FieldName, Float>> data) throws FileNotFoundException {

        // Init model from pmml file.
        ModelInvoker invoker = new ModelInvoker(new FileInputStream(pmmlModelPath));
        Map<String, Float> scores = new HashMap<String, Float>();

        // Predict
        for (Entry<String, Map<FieldName, Float>> d : data.entrySet()) {
            String key = d.getKey();
            Map<FieldName, Float> fieldData = d.getValue();

            Map<FieldName, ?> scoreMap = invoker.newinvoke(fieldData);
            Float score = Float.parseFloat(scoreMap.get(new FieldName("probability(1)")).toString());
            scores.put(key, score);
        }

        return scores;
    }

    /**
     * Read fieldname data from csv file.
     * @param filePath: test data path.
     * @return : map with row number as key.
     *
     */
    private static Map<String, Map<FieldName, Float>> fieldFromCsv(String filePath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String[] nameArr = br.readLine().split(",");
        Map<String, Map<FieldName, Float>> data = new HashMap<String, Map<FieldName, Float>>();

        // Read csv data to map.
        String paramLine = null;
        int numrow = 0;
        while ((paramLine = br.readLine()) != null) {
            String[] paramLineArr = paramLine.split(",");

            Map<FieldName, Float> map = new HashMap<FieldName, Float>();
            for (int i = 0; i < paramLineArr.length; i++) {
                map.put(FieldName.create(nameArr[i]), Float.parseFloat(paramLineArr[i]));
            }
            data.put(Integer.toString(numrow), map);
            numrow += 1;
        }

        return data;
    }

}
