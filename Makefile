# Zero DTE Options Strategy — Pipeline Makefile
# Usage:
#   make features        — build all features (no Theta Terminal required)
#   make target          — simulate IC outcomes and build target variable
#   make model_table     — join all features into a single daily table
#   make train           — train XGBoost model
#   make eval            — evaluate model performance
#   make score           — score today for live trading
#   make all             — full pipeline (features → target → model_table → train → eval)
#   make fetch           — fetch all data (requires Theta Terminal running)
#   make fetch_options   — fetch options data only (requires Theta Terminal)

PYTHON = python3
SCRIPTS = scripts
DATA = data
MODELS = models

# Environment: dev | uat | prod (default: dev)
# Usage: make score ENV=uat  |  make fetch ENV=prod
ENV ?= dev
ifeq ($(ENV),dev)
  ENV_FILE = .env.development
else ifeq ($(ENV),uat)
  ENV_FILE = .env.uat
else ifeq ($(ENV),prod)
  ENV_FILE = .env.production
else
  ENV_FILE = .env.development
endif

.PHONY: all features target model_table train eval score fetch fetch_options clean help

##──────────────────────────────────────────────────────────────────────────────
## Full pipeline
##──────────────────────────────────────────────────────────────────────────────

all: features target model_table train eval

##──────────────────────────────────────────────────────────────────────────────
## Feature engineering (no Theta Terminal needed — works on existing data)
##──────────────────────────────────────────────────────────────────────────────

features: $(DATA)/spy_features.parquet $(DATA)/options_features.parquet

$(DATA)/spy_merged.parquet: $(DATA)/spy_1min.parquet $(DATA)/vix_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_merged_table.py

$(DATA)/spy_features.parquet: $(DATA)/spy_merged.parquet
	$(PYTHON) $(SCRIPTS)/build_features.py

$(DATA)/options_features.parquet: $(DATA)/spxw_0dte_eod.parquet $(DATA)/spxw_0dte_oi.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_options_features.py

##──────────────────────────────────────────────────────────────────────────────
## Target variable (IC simulation)
##──────────────────────────────────────────────────────────────────────────────

target: $(DATA)/target.parquet

$(DATA)/target.parquet: $(DATA)/spxw_0dte_eod.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_target.py

##──────────────────────────────────────────────────────────────────────────────
## Model table (join everything → one row per trading day)
##──────────────────────────────────────────────────────────────────────────────

model_table: $(DATA)/model_table.parquet

$(DATA)/model_table.parquet: $(DATA)/spy_features.parquet $(DATA)/options_features.parquet $(DATA)/target.parquet
	$(PYTHON) $(SCRIPTS)/build_model_table.py

##──────────────────────────────────────────────────────────────────────────────
## Train XGBoost model
##──────────────────────────────────────────────────────────────────────────────

train: $(MODELS)/xgb_model.pkl

$(MODELS)/xgb_model.pkl: $(DATA)/model_table.parquet
	mkdir -p $(MODELS)
	$(PYTHON) $(SCRIPTS)/train_model.py

##──────────────────────────────────────────────────────────────────────────────
## Evaluate
##──────────────────────────────────────────────────────────────────────────────

eval: $(MODELS)/xgb_model.pkl
	$(PYTHON) $(SCRIPTS)/evaluate.py

##──────────────────────────────────────────────────────────────────────────────
## Score today (live trading)
##──────────────────────────────────────────────────────────────────────────────

score:
	@echo "Running score in [$(ENV)] mode ($(ENV_FILE))"
	@env $$(cat $(ENV_FILE) | grep -v '^#' | xargs) $(PYTHON) $(SCRIPTS)/score_live.py

score:uat:
	$(MAKE) score ENV=uat

score:prod:
	$(MAKE) score ENV=prod

##──────────────────────────────────────────────────────────────────────────────
## Data fetching (requires Theta Terminal on localhost:25503)
##──────────────────────────────────────────────────────────────────────────────

fetch: fetch_market fetch_options

fetch_market:
	$(PYTHON) $(SCRIPTS)/fetch_spx_daily.py
	$(PYTHON) $(SCRIPTS)/fetch_spx_data.py
	$(PYTHON) $(SCRIPTS)/fetch_vix_data.py
	$(PYTHON) $(SCRIPTS)/fetch_events_data.py
	$(PYTHON) $(SCRIPTS)/fetch_econ_calendar.py

fetch_options:
	$(PYTHON) $(SCRIPTS)/fetch_option_chain.py
	$(PYTHON) $(SCRIPTS)/fetch_term_structure.py
	$(PYTHON) $(SCRIPTS)/fetch_minute_quotes.py

##──────────────────────────────────────────────────────────────────────────────
## Cleanup
##──────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(DATA)/spy_merged.parquet
	rm -f $(DATA)/spy_features.parquet
	rm -f $(DATA)/options_features.parquet
	rm -f $(DATA)/target.parquet
	rm -f $(DATA)/model_table.parquet
	rm -f $(MODELS)/xgb_model.pkl

clean_all: clean
	rm -f $(DATA)/spxw_0dte_eod.parquet
	rm -f $(DATA)/spxw_0dte_oi.parquet
	rm -f $(DATA)/spxw_term_structure.parquet
	rm -f $(DATA)/spxw_0dte_minute_quotes.parquet

##──────────────────────────────────────────────────────────────────────────────
## Help
##──────────────────────────────────────────────────────────────────────────────

help:
	@echo "Zero DTE Options Strategy — Pipeline"
	@echo ""
	@echo "Commands:"
	@echo "  make features      Build SPY technical + options features"
	@echo "  make target        Simulate IC outcomes → target variable"
	@echo "  make model_table   Join all features into daily model table"
	@echo "  make train         Train XGBoost model"
	@echo "  make eval          Evaluate model (AUC, feature importance)"
	@echo "  make score         Score today for live trading"
	@echo "  make all           Full pipeline (features→target→train→eval)"
	@echo "  make fetch         Fetch all data (requires Theta Terminal)"
	@echo "  make fetch_options Fetch options data only (requires Theta Terminal)"
	@echo "  make clean         Remove generated files (keep raw data)"
	@echo ""
	@echo "Theta Terminal must be running on localhost:25503 for fetch commands."
