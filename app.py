from __future__ import annotations

import base64
import io
import math
import os
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".csv"}
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "20_000"))
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "10_000"))
MODEL_ESTIMATORS = int(os.environ.get("MODEL_ESTIMATORS", "25"))
MODEL_MAX_DEPTH = int(os.environ.get("MODEL_MAX_DEPTH", "10"))
ONE_HOT_MAX_CATEGORIES = int(os.environ.get("ONE_HOT_MAX_CATEGORIES", "12"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "128"))
RANDOM_STATE = 42

app = Flask(__name__)
app.config["SECRET_KEY"] = "big-csv-analyzer-dev-key"
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def upload_path(stored_name: str) -> Path:
    safe_name = secure_filename(stored_name)
    if safe_name != stored_name:
        raise ValueError("Некорректное имя файла")
    path = UPLOAD_DIR / safe_name
    if not path.exists():
        raise FileNotFoundError("Загруженный файл не найден")
    return path


def read_preview(path: Path, rows: int = 8) -> tuple[list[str], list[dict[str, Any]]]:
    preview = pd.read_csv(path, nrows=rows)
    return preview.columns.tolist(), preview.fillna("").to_dict(orient="records")


def update_numeric_stats(
    stats: dict[str, dict[str, float]],
    chunk: pd.DataFrame,
) -> None:
    numeric_chunk = chunk.select_dtypes(include=[np.number])
    for column in numeric_chunk.columns:
        values = pd.to_numeric(numeric_chunk[column], errors="coerce").dropna()
        if values.empty:
            continue

        column_stats = stats.setdefault(
            column,
            {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf},
        )
        count = float(values.count())
        column_stats["count"] += count
        column_stats["sum"] += float(values.sum())
        column_stats["sum_sq"] += float(np.square(values).sum())
        column_stats["min"] = min(column_stats["min"], float(values.min()))
        column_stats["max"] = max(column_stats["max"], float(values.max()))


def finalize_numeric_stats(stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, values in stats.items():
        count = values["count"]
        if count == 0:
            continue
        mean = values["sum"] / count
        variance = 0.0
        if count > 1:
            variance = (values["sum_sq"] - (values["sum"] ** 2 / count)) / (count - 1)
        rows.append(
            {
                "column": column,
                "count": int(count),
                "mean": round(mean, 4),
                "std": round(math.sqrt(max(variance, 0.0)), 4),
                "min": round(values["min"], 4),
                "max": round(values["max"], 4),
            }
        )
    return sorted(rows, key=lambda item: item["column"])


def analyze_csv(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    total_rows = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    missing_counts: pd.Series | None = None
    numeric_stats: dict[str, dict[str, float]] = {}
    sample_parts: list[pd.DataFrame] = []
    sample_rows = 0

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        if not columns:
            columns = chunk.columns.tolist()
            dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

        total_rows += len(chunk)
        current_missing = chunk.isna().sum()
        missing_counts = current_missing if missing_counts is None else missing_counts.add(current_missing, fill_value=0)
        update_numeric_stats(numeric_stats, chunk)

        if sample_rows < SAMPLE_LIMIT:
            remaining = SAMPLE_LIMIT - sample_rows
            sample_parts.append(chunk.head(remaining))
            sample_rows += min(len(chunk), remaining)

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(columns=columns)
    missing_counts = missing_counts if missing_counts is not None else pd.Series(dtype="int64")

    missing_summary = []
    for column in columns:
        count = int(missing_counts.get(column, 0))
        percent = round((count / total_rows * 100), 2) if total_rows else 0.0
        missing_summary.append({"column": column, "missing": count, "percent": percent})

    report = {
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "missing": missing_summary,
        "numeric_stats": finalize_numeric_stats(numeric_stats),
        "sample_rows": len(sample),
    }
    return report, sample


def infer_task_type(target: pd.Series) -> str:
    non_null = target.dropna()
    unique_count = non_null.nunique()
    if pd.api.types.is_numeric_dtype(non_null) and unique_count > 20 and unique_count / max(len(non_null), 1) > 0.05:
        return "regression"
    return "classification"


def build_pipeline(X: pd.DataFrame, task_type: str) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=ONE_HOT_MAX_CATEGORIES,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    model = (
        RandomForestRegressor(
            n_estimators=MODEL_ESTIMATORS,
            max_depth=MODEL_MAX_DEPTH,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        if task_type == "regression"
        else RandomForestClassifier(
            n_estimators=MODEL_ESTIMATORS,
            max_depth=MODEL_MAX_DEPTH,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced",
        )
    )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", model),
        ]
    )


def split_data(X: pd.DataFrame, y: pd.Series, task_type: str):
    stratify = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify = y

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def aggregate_feature_importance(pipeline: Pipeline, X: pd.DataFrame) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    transformed_names = preprocess.get_feature_names_out()
    scores = {column: 0.0 for column in X.columns}
    for transformed_name, importance in zip(transformed_names, importances):
        for column in X.columns:
            if transformed_name == f"num__{column}" or transformed_name.startswith(f"cat__{column}_"):
                scores[column] += float(importance)
                break

    return [
        {"feature": feature, "importance": round(score, 5)}
        for feature, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ][:15]


def train_model(sample: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in sample.columns:
        raise ValueError("Целевая колонка отсутствует в CSV")

    data = sample.dropna(subset=[target_column]).copy()
    if len(data) < 30:
        raise ValueError("Недостаточно строк с заполненной целевой колонкой для обучения модели")

    y = data[target_column]
    X = data.drop(columns=[target_column])
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("Нет признаков для обучения модели после удаления целевой колонки")

    task_type = infer_task_type(y)
    if task_type == "classification":
        y = y.astype(str)

    X_train, X_test, y_train, y_test = split_data(X, y, task_type)
    pipeline = build_pipeline(X_train, task_type)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics: dict[str, Any]
    if task_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        metrics = {"RMSE": round(float(rmse), 4), "R2": round(float(r2_score(y_test, predictions)), 4)}
    else:
        metrics = {"Accuracy": round(float(accuracy_score(y_test, predictions)), 4)}

    return {
        "task_type": "Регрессия" if task_type == "regression" else "Классификация",
        "target": target_column,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "feature_importance": aggregate_feature_importance(pipeline, X_train),
    }


def fig_to_base64() -> str:
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def target_distribution_chart(sample: pd.DataFrame, target_column: str, task_type_label: str) -> str | None:
    if target_column not in sample.columns:
        return None

    target = sample[target_column].dropna()
    if target.empty:
        return None

    plt.figure(figsize=(8, 4.2))
    if task_type_label == "Регрессия" and pd.api.types.is_numeric_dtype(target):
        plt.hist(target, bins=30, color="#287d8e", edgecolor="white")
        plt.xlabel(target_column)
        plt.ylabel("Количество")
    else:
        counts = target.astype(str).value_counts().head(20).sort_values()
        plt.barh(counts.index, counts.values, color="#287d8e")
        plt.xlabel("Количество")
        plt.ylabel(target_column)
    plt.title("Распределение целевой переменной")
    return fig_to_base64()


def correlation_chart(sample: pd.DataFrame) -> str | None:
    numeric = sample.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return None

    limited = numeric.iloc[:, :10]
    corr = limited.corr()
    plt.figure(figsize=(7.4, 6.2))
    image = plt.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title("Корреляция числовых признаков")
    return fig_to_base64()


def feature_importance_chart(feature_importance: list[dict[str, Any]]) -> str | None:
    if not feature_importance:
        return None

    items = list(reversed(feature_importance[:15]))
    plt.figure(figsize=(8, 4.8))
    plt.barh([item["feature"] for item in items], [item["importance"] for item in items], color="#d9822b")
    plt.xlabel("Важность")
    plt.title("Важность признаков модели")
    return fig_to_base64()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("dataset")
    if not uploaded_file or not uploaded_file.filename:
        flash("Выберите CSV-файл для загрузки.", "error")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Поддерживаются только CSV-файлы.", "error")
        return redirect(url_for("index"))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(uploaded_file.filename)
    stored_name = f"{uuid.uuid4().hex}_{original_name}"
    path = UPLOAD_DIR / stored_name
    uploaded_file.save(path)

    try:
        columns, preview = read_preview(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        flash(f"Не удалось прочитать CSV: {exc}", "error")
        return redirect(url_for("index"))

    return render_template(
        "select_target.html",
        stored_name=stored_name,
        original_name=uploaded_file.filename,
        columns=columns,
        preview=preview,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    stored_name = request.form.get("stored_name", "")
    target_column = request.form.get("target_column", "")

    try:
        path = upload_path(stored_name)
        report, sample = analyze_csv(path)
        model_report = train_model(sample, target_column)
    except Exception as exc:
        app.logger.exception("Analysis failed")
        flash(f"Ошибка анализа: {exc}", "error")
        return redirect(url_for("index"))

    charts = {
        "target_distribution": target_distribution_chart(sample, target_column, model_report["task_type"]),
        "correlation": correlation_chart(sample),
        "feature_importance": feature_importance_chart(model_report["feature_importance"]),
    }

    return render_template(
        "results.html",
        file_name=stored_name,
        report=report,
        model=model_report,
        charts=charts,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=debug)
