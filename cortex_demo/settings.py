IMAGE_SIZE = 16
INPUT_NEURONS = IMAGE_SIZE * IMAGE_SIZE
ORIENTATIONS = (0, 45, 90, 135)
NEURONS_PER_GROUP = 32

DEFAULT_DURATION_MS = 200
DEFAULT_DT_MS = 1.0
DEFAULT_MAX_RATE_HZ = 100.0
DEFAULT_SEED = 42

V_REST = -65.0
V_RESET = -68.0
V_THRESHOLD = -52.0
TAU_MEMBRANE_MS = 20.0
REFRACTORY_MS = 3
GAIN = 6.0
LATERAL_INHIBITION = 0.12

STIMULUS_LABELS = {
    "亮点": "spot",
    "水平条纹": "horizontal",
    "垂直条纹": "vertical",
    "左斜条纹": "diagonal_left",
    "右斜条纹": "diagonal_right",
    "空白": "blank",
}

