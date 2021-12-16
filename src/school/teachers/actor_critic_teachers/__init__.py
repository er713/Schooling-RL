from .learn_method import _learn_main
from .actor_critic_table_teacher import ActorCriticTableTeacher
from .actor_critic_n_last_teacher import ActorCriticNLastTeacher
from .actor_critic_all_history_cnn import ActorCriticAllHistoryTeacher

__all__ = ["ActorCriticNLastTeacher", "ActorCriticTableTeacher", "ActorCriticAllHistoryTeacher"]
