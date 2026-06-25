"""Run the Member 1 Telecom OSS anomaly detection pipeline."""

import pandas as pd

import anomaly_detector
import config
import data_loader
import feature_selection
import preprocessing


def main() -> None:
    dataset_path = config.DATA_DIR / "ds1_processed.csv"

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = data_loader.load_data(dataset_path)

    cleaned_data = preprocessing.clean_data(raw_data)
    cleaned_data = preprocessing.handle_missing_values(cleaned_data)

    preprocessing_pipeline, _ = preprocessing.build_preprocessing_pipeline(cleaned_data)
    processed_array = preprocessing_pipeline.fit_transform(cleaned_data)
    processed_feature_names = preprocessing_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    processed_data = pd.DataFrame(processed_array, columns=processed_feature_names)

    selected_data, selected_features, feature_report = feature_selection.select_features(
        processed_data
    )
    feature_report.to_csv(config.FEATURE_REPORT_PATH, index=False)

    model = anomaly_detector.train_model(selected_data)
    predictions = anomaly_detector.predict_anomalies(model, selected_data)
    anomaly_scores = anomaly_detector.compute_anomaly_scores(model, selected_data)

    record_ids = (
        cleaned_data["cell_id"]
        if "cell_id" in cleaned_data.columns
        else pd.Series(range(1, len(cleaned_data) + 1))
    )
    scores_df = anomaly_detector.scores_to_dataframe(
        anomaly_scores,
        predictions=predictions,
        record_ids=record_ids,
    )

    anomaly_scores_path = anomaly_detector.save_anomaly_scores(
        scores_df,
        config.ANOMALY_SCORES_PATH,
    )
    model_path = anomaly_detector.save_model(model, config.MODEL_PATH)

    anomaly_percentage = scores_df["is_anomaly"].mean() * 100

    print(f"Number of records: {len(raw_data)}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
    print("Output file locations:")
    print(f"- Anomaly scores: {anomaly_scores_path}")
    print(f"- Isolation Forest model: {model_path}")
    print(f"- Feature report: {config.FEATURE_REPORT_PATH}")


if __name__ == "__main__":
    main()
