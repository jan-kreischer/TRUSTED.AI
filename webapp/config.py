DEBUG = True # Turns on debugging features in Flask
BCRYPT_LOG_ROUNDS = 12 # Configuration for the Flask-Bcrypt extension
MAIL_FROM_EMAIL = "robert@example.com" # For use in application emails
SCENARIOS_FOLDER_PATH = "scenarios"
TRAINING_DATA_FILE_NAME_REGEX = "train.*"
TEST_DATA_FILE_NAME_REGEX = "test.*"
MODEL_REGEX = "model.*"
PICKLE_FILE_EXTENSIONS = [".sav", ".pkl", ".pickle"]
JOBLIB_FILE_EXTENSIONS = [".joblib"]

# Constants used for loading data from the solution
FACTSHEET_NAME = "factsheet.json"
SCENARIO_DESCRIPTION_FILE = "description.md"
SCENARIO_LINK_FILE = "link.md"

SOLUTIONS_FOLDER = "solutions"

# If no target column name is given, we assume 
# that the last column to the right is containing the label (or predicted value)
DEFAULT_TARGET_COLUMN_INDEX = -1

PRIMARY_COLOR = ''
SECONDARY_COLOR = ''
TRUST_COLOR = '#1a1a1a'
FAIRNESS_COLOR = '#06d6a0'
EXPLAINABILITY_COLOR = '#ffd166'
ROBUSTNESS_COLOR = '#ef476f'
METHODOLOGY_COLOR = '#118ab2'

# Paths
METRICS_CONFIG_PATH = "configs/metrics"
DEFAULT_METRICS_FILE ="default.json"
WEIGHTS_CONFIG_PATH = "configs/weights"
DEFAULT_WEIGHTS_FILE = "default.json"

XAXIS_TICKANGLE = 30





