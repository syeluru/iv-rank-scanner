# Zero DTE Options Strategy — Pipeline Makefile
# Usage:
#   make features        — build all features (no Theta Terminal required)
#   make fetch_macro     — fetch macro + cycle data from FRED (no Theta Terminal)
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

.PHONY: all features regime target model_table train eval score fetch fetch_options fetch_macro clean clean_all help

##──────────────────────────────────────────────────────────────────────────────
## Full pipeline
##──────────────────────────────────────────────────────────────────────────────

all: features target model_table train eval

##──────────────────────────────────────────────────────────────────────────────
## Feature engineering (no Theta Terminal needed — works on existing data)
##──────────────────────────────────────────────────────────────────────────────

features: $(DATA)/spx_features.parquet \
          $(DATA)/options_features.parquet \
          $(DATA)/iv_surface_features.parquet \
          $(DATA)/regime_features.parquet \
          $(DATA)/gex_regime_features.parquet \
          $(DATA)/vanna_charm_features.parquet \
          $(DATA)/momentum_features.parquet \
          $(DATA)/breadth_features.parquet \
          $(DATA)/cross_asset_features.parquet \
          $(DATA)/vol_expansion_features.parquet \
          $(DATA)/microstructure_features.parquet

# ── Core features (from Theta Terminal / existing raw data) ──

$(DATA)/spx_merged.parquet: $(DATA)/spx_1min.parquet $(DATA)/vix_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_merged_table.py

$(DATA)/spx_features.parquet: $(DATA)/spx_merged.parquet
	$(PYTHON) $(SCRIPTS)/build_features.py

$(DATA)/options_features.parquet: $(DATA)/spxw_0dte_eod.parquet $(DATA)/spxw_0dte_oi.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_options_features.py

$(DATA)/regime_features.parquet: $(DATA)/spx_daily.parquet $(DATA)/vix_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_regime_features.py

$(DATA)/iv_surface_features.parquet: $(DATA)/spxw_0dte_intraday_greeks.parquet $(DATA)/spxw_term_structure.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_iv_surface_features.py

$(DATA)/gex_regime_features.parquet: $(DATA)/options_features.parquet $(DATA)/vix_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_gex_regime_features.py

$(DATA)/vanna_charm_features.parquet: $(DATA)/spxw_0dte_intraday_greeks.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_vanna_charm_features.py

$(DATA)/momentum_features.parquet: $(DATA)/spx_merged.parquet
	$(PYTHON) $(SCRIPTS)/build_momentum_features.py

# ── New feature modules (no Theta Terminal — yfinance or pure calculation) ──

$(DATA)/breadth_features.parquet:
	$(PYTHON) $(SCRIPTS)/build_breadth_features.py

$(DATA)/cross_asset_features.parquet:
	$(PYTHON) $(SCRIPTS)/build_cross_asset_features.py

$(DATA)/vol_expansion_features.parquet: $(DATA)/vix_daily.parquet $(DATA)/spx_daily.parquet
	$(PYTHON) $(SCRIPTS)/build_vol_expansion_features.py

$(DATA)/microstructure_features.parquet: $(DATA)/target.parquet $(DATA)/spx_features.parquet $(DATA)/momentum_features.parquet
	$(PYTHON) $(SCRIPTS)/build_microstructure_features.py

##──────────────────────────────────────────────────────────────────────────────
## Target variable (1-min IC simulation — one row per minute per day)
##──────────────────────────────────────────────────────────────────────────────

target: $(DATA)/target.parquet

$(DATA)/target.parquet: $(DATA)/spx_1min.parquet
	$(PYTHON) $(SCRIPTS)/build_target.py

##──────────────────────────────────────────────────────────────────────────────
## Model table (join features + target → one row per minute per day)
##──────────────────────────────────────────────────────────────────────────────

model_table: $(DATA)/model_table.parquet

$(DATA)/model_table.parquet: $(DATA)/spx_features.parquet \
                             $(DATA)/options_features.parquet \
                             $(DATA)/iv_surface_features.parquet \
                             $(DATA)/regime_features.parquet \
                             $(DATA)/gex_regime_features.parquet \
                             $(DATA)/target.parquet
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

score-uat:
	$(MAKE) score ENV=uat

score-prod:
	$(MAKE) score ENV=prod

##──────────────────────────────────────────────────────────────────────────────
## Data fetching (requires Theta Terminal on localhost:25503)
##──────────────────────────────────────────────────────────────────────────────

fetch: fetch_market fetch_options fetch_macro

fetch_market:
	$(PYTHON) $(SCRIPTS)/fetch_spx_intraday.py
	$(PYTHON) $(SCRIPTS)/fetch_vix_data.py
	$(PYTHON) $(SCRIPTS)/fetch_vix1d.py
	$(PYTHON) $(SCRIPTS)/fetch_events_data.py
	$(PYTHON) $(SCRIPTS)/fetch_econ_calendar.py

fetch_options:
	$(PYTHON) $(SCRIPTS)/fetch_option_chain.py
	$(PYTHON) $(SCRIPTS)/fetch_term_structure.py

# Macro + cycle data (no Theta Terminal required — FRED API + yfinance)
fetch_macro:
	$(PYTHON) $(SCRIPTS)/fetch_macro_features.py
	$(PYTHON) $(SCRIPTS)/fetch_cycle_features.py
	$(PYTHON) $(SCRIPTS)/fetch_vix1d.py

##──────────────────────────────────────────────────────────────────────────────
## Cleanup
##──────────────────────────────────────────────────────────────────────────────

clean:
	rm -f $(DATA)/spx_merged.parquet
	rm -f $(DATA)/spx_features.parquet
	rm -f $(DATA)/options_features.parquet
	rm -f $(DATA)/target.parquet
	rm -f $(DATA)/model_table.parquet
	rm -f $(DATA)/breadth_features.parquet
	rm -f $(DATA)/cross_asset_features.parquet
	rm -f $(DATA)/vol_expansion_features.parquet
	rm -f $(DATA)/microstructure_features.parquet
	rm -f $(MODELS)/xgb_model.pkl

clean_all: clean
	rm -f $(DATA)/spxw_0dte_eod.parquet
	rm -f $(DATA)/spxw_0dte_oi.parquet
	rm -f $(DATA)/spxw_term_structure.parquet
	rm -f $(DATA)/spxw_0dte_minute_quotes.parquet
	rm -f $(DATA)/macro_regime.parquet
	rm -f $(DATA)/presidential_cycles.parquet
	rm -f $(DATA)/vix1d_daily.parquet

##──────────────────────────────────────────────────────────────────────────────
## Help
##──────────────────────────────────────────────────────────────────────────────

help:
	@echo "Zero DTE Options Strategy — Pipeline"
	@echo ""
	@echo "Commands:"
	@echo "  make features      Build all features (SPX, options, macro, breadth, cross-asset, etc.)"
	@echo "  make target        Simulate IC outcomes → target variable"
	@echo "  make model_table   Join all features into daily model table"
	@echo "  make train         Train XGBoost model"
	@echo "  make eval          Evaluate model (AUC, feature importance)"
	@echo "  make score         Score today for live trading"
	@echo "  make all           Full pipeline (features→target→train→eval)"
	@echo "  make fetch         Fetch all data (Theta Terminal + FRED + yfinance)"
	@echo "  make fetch_macro   Fetch macro + cycle data only (no Theta Terminal)"
	@echo "  make fetch_options Fetch options data only (requires Theta Terminal)"
	@echo "  make clean         Remove generated feature/model files (keep raw data)"
	@echo ""
	@echo "Theta Terminal must be running on localhost:25503 for fetch/fetch_options."
	@echo "fetch_macro requires FRED_API_KEY in .env.development."
