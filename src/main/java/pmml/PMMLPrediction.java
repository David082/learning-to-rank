package pmml;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/*
 * Created by yu_wei on 2018/5/3.
 *
 * xgboost模型通过pmml存储，在java中调用
 * https://blog.csdn.net/luoyexuge/article/details/80094265
 *
 * lightgbm模型通过pmml存储，在java中调用
 * https://blog.csdn.net/luoyexuge/article/details/80087952
 *
 */
public class PMMLPrediction {

    public static void main(String[] args) throws Exception {

        String pathxml = "resources/pmml/dt.pmml";
        Map<String, Double> map = new HashMap<String, Double>();
        map.put("sepal_length", 5.1);
        map.put("sepal_width", 3.5);
        map.put("petal_length", 1.4);
        map.put("petal_width", 0.2);
        predictLrHeart(map, pathxml);

    }

    private static void predictLrHeart(Map<String, Double> irismap, String pathxml) throws Exception {

        PMML pmml;
        // 模型导入
        File file = new File(pathxml);
        try (InputStream is = new FileInputStream(file)) {
            pmml = PMMLUtil.unmarshal(is);

            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
            ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
            Evaluator evaluator = (Evaluator) modelEvaluator;

            List<InputField> inputFields = evaluator.getInputFields();
            // 过模型的原始特征，从画像中获取数据，作为模型输入
            Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
            for (InputField inputField : inputFields) {
                FieldName inputFieldName = inputField.getName();
                Object rawValue = irismap.get(inputFieldName.getValue());
                FieldValue inputFieldValue = inputField.prepare(rawValue);
                arguments.put(inputFieldName, inputFieldValue);
            }

            Map<FieldName, ?> results = evaluator.evaluate(arguments);
            List<TargetField> targetFields = evaluator.getTargetFields();
            //对于分类问题等有多个输出。
            for (TargetField targetField : targetFields) {
                FieldName targetFieldName = targetField.getName();
                Object targetFieldValue = results.get(targetFieldName);
                System.err.println("target: " + targetFieldName.getValue() + " value: " + targetFieldValue);
            }
        }
    }

}
