import json
from pathlib import Path

def save_evaluation_results(path_save, args, model_config, score):
    """
    Save evaluation results to a JSON file using args and model_config.

    Args:
        path_save (str or Path): Directory to save the results.
        args (argparse.Namespace): Parsed command-line arguments.
        model_config (dict): Model configuration dictionary.
        score (float or dict): Evaluation score to save.
    """
    save_dir = Path(path_save)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / 'evaluation_results.json'

    results = {
        'model_name': args.model_name,
        'pretrained': model_config.get('pretrained', False),
        'blurr_level': args.blurr_level,
        'metric': args.metric_name,
        'decoding_method': args.decoding_method,
        'score': score
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved evaluation results to: {output_file}")
