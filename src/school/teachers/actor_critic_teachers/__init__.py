from .learn_method import _learn_main
from .actor_critic_table_teacher import ActorCriticTableTeacher
from .actor_critic_n_last_teacher import ActorCriticNLastTeacher
from .actor_critic_all_history_cnn import ActorCriticAllHistoryCNNTeacher

__all__ = ["ActorCriticNLastTeacher", "ActorCriticTableTeacher", "ActorCriticAllHistoryCNNTeacher"]
