.PHONY: test export-notebooks reproduce

test:
	python -m pytest -q

# usage: make export-notebooks NB_DIR=path/to/downloaded_kaggle_ipynb_files
export-notebooks:
	python scripts/export_notebooks.py $(NB_DIR)

# The data and models are built in Kaggle notebooks (see README.md). Run them in this order:
reproduce:
	@echo "1. scripts/download_data.sh  (or upload the two csv files as a Kaggle dataset)"
	@echo "2. notebooks/layer3a_data_features_eda.py      -> features, splits, EDA"
	@echo "3. notebooks/layer3a_models_calibration.py     -> models, calibration, serving retrain cells R1-R6"
	@echo "4. notebooks/layer3a_core_vs_full.py, layer3a_shap_conformal.py, layer3a_serving_check.py"
	@echo "5. notebooks/layer3a_serving_parity.py         -> serving code + parity check"
	@echo "6. notebooks/layer3a_final_bundle.py           -> artifact bundle, reports, final test evaluation"
