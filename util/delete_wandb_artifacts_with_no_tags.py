import wandb

if __name__ == "__main__":

    """
    deletes all models that do not have a tag attached

    by default this means wandb will delete all but the "latest" or "best" models

    set dry_run == False to delete...
    """

    dry_run = False
    api = wandb.Api(overrides={"project": "mamba-runs"})
    project = api.project("mamba-runs")

    for artifact_type in project.artifacts_types():
        for artifact_collection in artifact_type.collections():
            for version in api.artifacts(artifact_type.type, artifact_collection.name):
                if artifact_type.type == "model":
                    if len(version.aliases) > 0:
                        # print out the name of the one we are keeping
                        print(f"KEEPING {version.name}")
                    else:
                        print(f"DELETING {version.name}")
                        if not dry_run:
                            version.delete()
