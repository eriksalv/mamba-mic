if __name__ == "__main__":
    import wandb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()

    run = wandb.init(project="artifacts", job_type="upload-model")
    artifact = wandb.Artifact(name=f"{args.model_name}_artifact", type="model")
    artifact.add_file(
        local_path=args.ckpt_path,
        name=f"{args.model_name}_{args.ckpt_path.split('/')[-1]}",
    )
    artifact.save()
    wandb.finish()
