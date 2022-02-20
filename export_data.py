import wandb
import pandas as pd

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("eryk/schooling-rl")  # TODO zmienić na użytkownik/nazwa_projektu
# summary_list = []
# config_list = []
# name_list = []
al_pd = []
# next(runs)
for run in runs:
    if run.name.endswith("ppo_test"):  # TODO jakiś warunek po jakich nazwach wybierać, pewnie "benchmark"
        print(run.id)
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        # summary_list.append(run.summary._json_dict)
        hist: pd.DataFrame = run.history()
        # print(len(hist))
        hl = len(hist)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        # config_list.append(config)
        c2 = {k: [v]*hl for k, v in config.items()}
        # print(c2)
        con_pd = pd.DataFrame.from_dict(c2)
        # print(con_pd.to_string())
        al = pd.concat([hist, con_pd], axis=1)
        # print(al.to_string())
        al_pd.append(al)
        # print(config.keys())
        # print(config["skills_quantity"], config["train_time_per_skill"], config["env_name"])
        # run.name is the name of the run.
        # name_list.append(run.name)
        # break

all_df = pd.concat(al_pd).reset_index()

all_df.to_csv("history.csv")
