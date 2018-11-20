package pyxgboost;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

/*
 * Created by yu_wei on 2018/11/20.
 */
public class LoadModel {

    public static void main(String[] args) throws XGBoostError {
        Booster booster = XGBoost.loadModel("resources/xgb_sample.model");
        System.out.println(booster);
    }
}
