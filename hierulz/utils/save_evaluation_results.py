def save_evaluation_results(path_save, args, model_config, score):
    """
    Save evaluation results to a JSON file by appending new results, without overwriting existing ones.

    Handles existing files that are either a dict (legacy) or a list of dicts.
    """
    from pathlib import Path
    import json

    save_dir = Path(path_save)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / 'evaluation_results.json'

    new_result = {
        'model_name': args.model_name,
        'pretrained': model_config.get('pretrained', False),
        'blurr_level': args.blurr_level,
        'metric': args.metric_name,
        'decoding_method': args.decoding_method,
        'score': score
    }

    # Load and normalize existing results
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing = json.load(f)
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                raise ValueError("Invalid format in evaluation_results.json")
    else:
        existing = []

    existing.append(new_result)

    with open(output_file, 'w') as f:
        json.dump(existing, f, indent=4)

    print(f"Appended result to {output_file}")
