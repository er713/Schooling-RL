import wandb
import pandas as pd

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("eryk/schooling-rl")  # TODO change to <entity/project-name>
runs_df = []

for run in runs:
    if run.name == "benchmark":  # TODO change if want to choose runs with other name than "benchmark"
        print(run.id)
        history_df: pd.DataFrame = run.history()
        history_len = len(history_df)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}

        config_expanded = {k: [v] * history_len for k, v in config.items()}
        con_pd = pd.DataFrame.from_dict(config_expanded)

        run_df = pd.concat([history_df, con_pd], axis=1)
        runs_df.append(run_df)

all_df = pd.concat(runs_df).reset_index()

all_df.to_csv("history.csv")
