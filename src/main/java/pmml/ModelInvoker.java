package pmml;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public class ModelInvoker {
	private ModelEvaluator<? extends Model> modelEvaluator;

	// private final ModelEvaluator modelEvaluator;
	public ModelInvoker () {}

	public ModelInvoker(String pmmlFileName) {
		PMML pmml = null;
		InputStream is = null;
		try {
			if (pmmlFileName != null) {
				is = ModelInvoker.class.getClassLoader().getResourceAsStream(pmmlFileName);
				pmml = PMMLUtil.unmarshal(is);
			}

			try {
				is.close();
			} catch (IOException localIOException) {
			}

			this.modelEvaluator = ModelEvaluatorFactory.newInstance().newModelEvaluator(pmml);
		} catch (SAXException e) {
			pmml = null;
		} catch (JAXBException e) {
			pmml = null;
		} finally {
			try {
				is.close();
			} catch (IOException localIOException3) {
			}
		}

		this.modelEvaluator.verify();
	}

	public ModelInvoker(InputStream is) {
		PMML pmml = null;
		try {
			pmml = PMMLUtil.unmarshal(is);

			try {
				is.close();
			} catch (IOException localIOException) {
			}

			this.modelEvaluator = ModelEvaluatorFactory.newInstance().newModelEvaluator(pmml);

		} catch (SAXException e) {
			pmml = null;
		} catch (JAXBException e) {
			pmml = null;
		} finally {
			try {
				is.close();
			} catch (IOException localIOException3) {
			}
		}

		this.modelEvaluator.verify();
	}

	public Map<FieldName, ?> invoke(Map<FieldName, Object> paramsMap) {
		return this.modelEvaluator.evaluate(paramsMap);
	}

	public Map<FieldName, ?> newinvoke(Map<FieldName, Float> paramsMap) {
		return this.modelEvaluator.evaluate(paramsMap);
	}
}
