from pathlib import Path


# project paths
BASE_DIR = Path(__file__).resolve().parent.parent

# check local directory first, then parent directory if running inside a subfolder
raw_dir = BASE_DIR / "TCGA-PANCAN-HiSeq-801x20531"
if not raw_dir.exists() and (BASE_DIR.parent / "TCGA-PANCAN-HiSeq-801x20531").exists():
    raw_dir = BASE_DIR.parent / "TCGA-PANCAN-HiSeq-801x20531"

RAW_DATA_PATH = raw_dir / "data.csv"
RAW_LABELS_PATH = raw_dir / "labels.csv"

# directories for saved data and outputs
DATA_DIR = BASE_DIR / "data"
CACHE_NPZ_PATH = DATA_DIR / "data_cache.npz"
SPLITS_PATH = DATA_DIR / "splits_stratified_10repeats.npz"
N_EFFICIENCY_SPLITS_PATH = DATA_DIR / "splits_n_efficiency.npz"

RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# create directories if they don't exist yet
for d in [DATA_DIR, RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# dataset metadata
CLASS_NAMES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
NUM_CLASSES = 5
TOTAL_SAMPLES = 801
TOTAL_GENES = 20531


# experiment parameters
RANDOM_STATE = 42
N_SPLITS = 10
TEST_SIZE = 0.20

DEFAULT_P = 1000
P_GRID = [50, 100, 500, 1000, 5000, 20531]
N_TRAIN_GRID = [30, 60, 120, 250, 640]
TOP_K_STABILITY = 50
