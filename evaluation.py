from comet import download_model, load_from_checkpoint
import os
import certifi

def evaluate_comet_in_batches(sources, references, candidates, comet_model, batch_size, gpus):
    scores = []
    # Prepare batches
    for i in range(0, len(sources), batch_size):
        print(f'Processing batch {i + 1} to {min(i + batch_size, len(sources))} out of {len(sources)}')
        batch_samples = [{
            'src': sources[j],
            'mt': candidates[j],
            'ref': references[j]
        } for j in range(i, min(i + batch_size, len(sources)))]

        # Predict scores for the batch
        batch_scores = comet_model.predict(batch_samples, batch_size=batch_size, gpus=gpus)
        # Extract segment-level scores from the prediction object
        scores.extend(batch_scores.scores)
    return scores


def main():
    os.environ['SSL_CERT_FILE'] = certifi.where()

    # Download and load the COMET model
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    source_dir = 'source_files'
    reference_dir = 'reference_files'
    output_dir = 'output_files'
    eval_dir = 'evaluations'

    source_files = sorted(os.listdir(source_dir))
    reference_files = sorted(os.listdir(reference_dir))[1:]  # Adjust as needed
    output_files = sorted(os.listdir(output_dir))

    # Define batch size and GPUs
    batch_size = 128  # Adjust based on your system's memory and the size of your data
    gpus = 1  # Set to the number of GPUs available, or 0 if none

    for sf, rf, of in zip(source_files, reference_files, output_files):
        print(f'Evaluating {sf} with {rf} and {of}')
        with open(f'{source_dir}/{sf}', 'r') as f:
            source = f.read().splitlines()
        with open(f'{reference_dir}/{rf}', 'r') as f:
            reference = f.read().splitlines()
        with open(f'{output_dir}/{of}', 'r') as f:
            output = f.read().splitlines()

        total_scores = evaluate_comet_in_batches(source, reference, output, comet_model, batch_size, gpus)
        with open(f'{eval_dir}/{sf}_evaluation.txt', 'w') as f:
            f.write('\n'.join(map(str, total_scores)))

if __name__ == '__main__':
    main()
