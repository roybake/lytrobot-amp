from sensor_training.networks import SensorActorCritic, RecurrentSensorActorExport
from .config import SENSOR_FRAME_DIM, RECURRENT_HIDDEN_DIM

def build_policy(action_dim=14, privileged_context_dim=0):
    # GRU gets sensor frame + previous action by the wrapper's explicit controller buffer.
    return SensorActorCritic(obs_dim=SENSOR_FRAME_DIM, action_dim=action_dim,
        sensor_frame_dim=SENSOR_FRAME_DIM, actor_history_frames=1,
        recurrent_hidden_dim=RECURRENT_HIDDEN_DIM, privileged_context_dim=privileged_context_dim)
