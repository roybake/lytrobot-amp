from lytrobot_amp.foot_force import FootForceSensorConfig, FOOT_FORCE_NAMES

SENSOR_FRAME_DIM = 62
RECURRENT_INPUT_DIM = 76  # current 62D frame + controller-known previous action
RECURRENT_HIDDEN_DIM = 128
