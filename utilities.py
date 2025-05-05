import pandas as pd
import pandasql 
from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text as sql_text
from datetime import date 

from dotenv import load_dotenv
load_dotenv()
import os 
connection_url = URL.create(drivername='mysql',
                            username=os.getenv("DB_USERNAME"),
                            password=os.getenv("DB_PASSWORD"),
                            host=os.getenv("DB_HOST"),
                            database=os.getenv("DB_NAME"),
                            query={"ssl_ca": os.getenv("CA_CERT_PATH")})
engine = create_engine(connection_url)

def dataframe_to_sql( dataframe , table_name, version, new_table = 'NO'):
    if new_table == 'NO':
        dataframe['version'] = version
        dataframe['datetime_stamp'] = str(date.today())
        dataframe.to_sql(table_name, con = engine, if_exists = 'append', index = False)  
    else:
        dataframe['version'] = version
        dataframe['datetime_stamp'] = str(date.today())
        dataframe.to_sql(table_name, con = engine, if_exists = 'replace',index = False)  





from imblearn.pipeline import Pipeline as imPipeline
import os
import yaml
import joblib
import importlib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as imPipeline

def generate_dummy_data():
    X, y = make_classification(n_samples=500000, n_features=10, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    df['age'] = np.random.randint(20, 60, size=df.shape[0])
    df['income'] = np.random.randint(30000, 100000, size=df.shape[0])
    return df

# ---------- OUTLIER REMOVER ----------
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower = Q1 - self.factor * IQR
        self.upper = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        return X.where((X >= self.lower) & (X <= self.upper), other=np.nan).fillna(X.mean())

# ---------- DYNAMIC IMPORT ----------
def dynamic_import(path):
    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {path}: {e}")

class MLExperiment:
    def __init__(self, config, global_config, data, output_dir="model_outputs"):
        self.cfg = config
        self.global_cfg = global_config
        self.data = data.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.iteration_name = config["iteration_name"]
        self.model_class = self._get_model_class(config["model"])
        self.hyperparameters = config["hyperparameters"]
        self.scaler = self._get_scaler(config["scaler"], global_config["scalers"])
        self.sampler = self._get_sampler(config["sampler"], global_config["samplers"])
        self.feature_set = config["feature_set"]
        self.outlier_treatment = config.get("outlier_treatment", False)
        self.filter_condition = config.get("filter_condition")
        self.comments = config.get("comments", "")
        self.scoring = global_config.get("scoring", {"accuracy": "accuracy"})
        self.refit = config.get("refit", list(self.scoring.keys())[0])

    def _get_model_class(self, name):
        if name == "RandomForestClassifier":
            return RandomForestClassifier
        elif name == "LogisticRegression":
            return LogisticRegression
        else:
            raise ValueError(f"Unsupported model: {name}")

    def _get_scaler(self, scaler_key, scaler_map):
        ScalerClass = dynamic_import(scaler_map[scaler_key])
        return ScalerClass()

    def _get_sampler(self, sampler_key, sampler_map):
        if sampler_key:
            SamplerClass = dynamic_import(sampler_map[sampler_key])
            return SamplerClass()
        return None

    def preprocess(self):
        data = self.data.copy()
        if self.filter_condition:
            data = data.query(self.filter_condition)
        if self.feature_set:
            data = data[self.global_cfg['feature_sets'][self.feature_set] + [self.global_cfg["target_column"]]]
        return data

    def train(self):
        data = self.preprocess()
        target_col = self.global_cfg["target_column"]
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.global_cfg.get("test_size", 0.2),
            random_state=self.global_cfg.get("random_state", 42)
        )

        steps = []
        if self.outlier_treatment:
            outlier_remover = OutlierRemover()
            outlier_remover.fit(X_train)
            steps.append(("outlier", outlier_remover))
        steps.append(("scaler", self.scaler))

        if self.sampler:
            steps.append(("sampler", self.sampler))

        steps.append(("model", self.model_class()))
        pipeline = imPipeline(steps)

        param_grid = {f"model__{k}": v for k, v in self.hyperparameters.items()}
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            refit=self.refit,
            cv=5,
            return_train_score=True,
            verbose = 10,
            n_jobs = -1
        )
        grid.fit(X_train, y_train)

        cv_results = grid.cv_results_
        num_folds = len([k for k in cv_results if k.startswith("split0_train_")])
        num_configs = len(cv_results["params"])
        all_fold_rows = []

        # Log results for each hyperparameter combination and each fold
        for i in range(num_configs):
            print('reached here')
            params = cv_results["params"][i]
            # For each fold in the cross-validation
            for fold in range(num_folds):  # Assuming cv=5
                row = {
                    "iteration_name": self.iteration_name,
                    "model": self.cfg["model"],
                    "scaler": self.cfg["scaler"],
                    "params": params,
                    "fold": fold,
                    "outlier_treatment": self.outlier_treatment,
                    "oversampler": self.sampler is not None,
                    "sampler_params": str(self.sampler.get_params()) if self.sampler else "None",
                    "feature_set": self.feature_set,
                    "comments": self.comments
                }
                for metric in self.scoring.keys():
                    row[f"train_{metric}"] = cv_results[f"split{fold}_train_{metric}"][i]
                    row[f"val_{metric}"] = cv_results[f"split{fold}_test_{metric}"][i]
                all_fold_rows.append(row)

            # Evaluate the test set for this combination of hyperparameters
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            test_preds = pipeline.predict(X_test)
            test_row = {
                "iteration_name": self.iteration_name,
                "model": self.cfg["model"],
                "scaler": self.cfg["scaler"],
                "params": params,
                "fold": -1,  # Indicating this is the test set evaluation
                "outlier_treatment": self.outlier_treatment,
                "oversampler": self.sampler is not None,
                "sampler_params": str(self.sampler.get_params()) if self.sampler else "None",
                "feature_set": self.feature_set,
                "comments": self.comments
            }

            # Calculate and store the test scores for each metric
            for metric_name, scorer in {
                "accuracy": accuracy_score,
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score
            }.items():
                if metric_name in self.scoring:
                    test_row[f"test_{metric_name}"] = scorer(y_test, test_preds)
            all_fold_rows.append(test_row)

        # Save model and results
        # joblib.dump(grid.best_estimator_, os.path.join(self.output_dir, f"{self.iteration_name}_model.pkl"))
        return pd.DataFrame(all_fold_rows)

# ---------- TESTING THE WHOLE PROCESS ----------
# Load config
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# Dummy data
data = generate_dummy_data()

# Run experiments
results = []
for experiment_cfg in config["experiments"]:
    experiment = MLExperiment(experiment_cfg, config["global"], data)
    result_df = experiment.train()
    results.append(result_df)

# Combine and save results
final_results = pd.concat(results, ignore_index=True)
final_results.to_csv("experiment_results.csv", index=False)
