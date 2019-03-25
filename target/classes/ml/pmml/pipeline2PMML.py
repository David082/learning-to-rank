# -*- coding: utf-8 -*-
"""
Created on 2018-05-08 9:36

refer :
-- java version
https://stackoverflow.com/questions/29697543/registry-key-error-java-version-has-value-1-8-but-1-7-is-required
"""
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import collections
import numpy as np
from collections import defaultdict


class HourOfDayTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        hours = pd.DataFrame(X['datetime'].apply(lambda x: x.hour))
        return hours

    def fit(self, X, y=None, **fit_params):
        return self


class DistLimit(BaseEstimator, TransformerMixin):
    """ compute the distance score :
            if dist < 100: score1
            elif dist < 40000: score2
            else: score3"""

    def __init__(self, params):
        self.dist1 = params["dist1"]
        self.dist2 = params["dist2"]
        self.score1 = params["score1"]
        self.score2 = params["score2"]
        self.score3 = params["score3"]
        self.dist_index = params["dist_index"]
        self.columns = params["columns"]

    def transform(self, X):
        def dist_score_com(dist_val):
            result = 0.0
            if dist_val <= self.dist1:
                result = self.score1
            elif dist_val <= self.dist2:
                result = self.score2
            elif dist_val > self.dist2:
                result = self.score3
            return result

        X_copy = pd.DataFrame(X, columns=self.columns)
        _, self.feature_len = X_copy.shape
        assert self.feature_len == 208
        X_copy["score_dist"] = X_copy.apply(lambda x: dist_score_com(x[self.columns[self.dist_index]]), axis=1)
        return X_copy

    def fit(self, X, y=None):
        """ space """
        return self


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """A tranformer for combining low count observations for categorical features.

    This transformer will preserve category values that are above a certain
    threshold, while bucketing together all the other values. This will fix issues
    where new data may have an unobserved category value that the training data
    did not have.
    """

    def __init__(self, threshold=0.05):
        """Initialize method.

        Args:
            threshold (float): The threshold to apply the bucketing when
                categorical values drop below that threshold.
        """
        self.d = collections.defaultdict(list)
        self.threshold = threshold

    def transform(self, X, **transform_params):
        """Transforms X with new buckets.

        Args:
            X (obj): The dataset to pass to the transformer.

        Returns:
            The transformed X with grouped buckets.
        """
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].apply(lambda x: x if x in self.d[col] else 'CategoryGrouperOther')
        return X_copy

    def fit(self, X, y=None, **fit_params):
        """Fits transformer over X.

        Builds a dictionary of lists where the lists are category values of the
        column key for preserving, since they meet the threshold.
        """
        df_rows = len(X.index)
        for col in X.columns:
            calc_col = X.groupby(col)[col].agg(lambda x: (len(x) * 1.0) / df_rows)
            self.d[col] = calc_col[calc_col >= self.threshold].index.tolist()
        return self


class LookupTransformer(TransformerMixin):
    def __init__(self, base_dict, default_value=0.0):
        self.base_dict = base_dict
        self.default_value = default_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transform_dict = defaultdict(lambda: self.default_value)
        transform_dict.update(self.base_dict)
        return np.array(X.apply(lambda x: transform_dict[x]))


if __name__ == '__main__':
    print("Hello")
    filePath = "2018POI_check2_online.csv"
    data = pd.read_csv(filePath)
    print(data.head())
    # data.head()
    columns = data.columns.tolist()
    dist_index = columns.index("dist")
    params = {"dist1": 100, "dist2": 40000,
              "score1": 0.1, "score2": 0.2, "score3": 0.3,
              "columns": columns, "dist_index": dist_index}
    # dl = DistLimit(params)
    # dl.fit(data.values,None)
    # tmp = dl.transform(data.values)
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    # pipe = make_pipeline(MyANNTransformer(ann),
    #                 SVC())
    # pp_lr = Pipeline(DistLimit())
    # pp_lr = make_pipeline(DistLimit(params))
    # pp_lr.fit(data.values)
    # tmp = pp_lr.transform(data.values)

    from sklearn2pmml import PMMLPipeline
    from sklearn2pmml import sklearn2pmml

    # ------------------------------------------------- LogisticRegression
    iris_pipeline = PMMLPipeline([("classifier", DistLimit(params))])
    iris_pipeline.fit(data.values)
    # sklearn2pmml(iris_pipeline, "DecisionTreeIris.pmml", with_repr = True)
    # sklearn2pmml(iris_pipeline, "pp_lr.pmml", with_repr = True)
    sklearn2pmml(iris_pipeline, 'test_pp_lr.pmml', with_repr=True)

    # # ------------------------------------------------- LogisticRegression
    # from sklearn.linear_model import LogisticRegression
    #
    # iris_df = pd.read_csv("Iris.csv")
    # iris_pipeline = PMMLPipeline([
    #     # ("mapper", DataFrameMapper([
    #     #     (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), Imputer()])
    #     # ])),
    #     # ("pca", PCA(n_components=3)),
    #     # ("selector", SelectKBest(k=2)),
    #     ("classifier", LogisticRegression())
    # ])
    # # iris_pipeline.fit(iris_df, iris_df["Species"])
    # iris_pipeline.fit(iris_df[iris_df.columns.difference(["Species"])], iris_df["Species"])
    # sklearn2pmml(iris_pipeline, "LogisticRegressionIris.pmml", with_repr=True)
