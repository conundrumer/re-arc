import tqdm
import os
import json
import csv
import inspect

from lib import *
from main import get_verifiers

INVARIANCE_ATTEMPTS = 10
def generate_task_metadata():

    verifiers_mapper = get_verifiers()
    for task_id, verifier in verifiers_mapper.items():

        # Get num_steps: count num assignments (=)
        verifier_code = inspect.getsource(verifier)
        num_steps = verifier_code.count('=') - verifier_code.count('==') - verifier_code.count('!=') - verifier_code.count('<=') - verifier_code.count('>=')

        # Get task json in arc_original/training
        task_json_path = f'arc_original/training/{task_id}.json'
        with open(task_json_path, 'r') as f:
            task_data = json.load(f)

        # Count num train/test
        num_train = len(task_data['train'])
        num_test = len(task_data['test'])

        # Check color invariance
        # Sample generations and count how many DON'T break the verifier
        color_failures = 0
        for _ in tqdm.trange(INVARIANCE_ATTEMPTS, desc=f'{task_id}: color attempts', leave=False):
            try:
                generate_examples(task_id, num_examples=num_train + num_test, color_transform=True)
                # Success means we found a transform that breaks verifier
            except TransformError:
                # Couldn't find a transform that breaks verifier -> transform preserves it
                color_failures += 1

        # Invariance = proportion of transforms that DON'T break the verifier
        color_invariance = color_failures / INVARIANCE_ATTEMPTS

        # Check dihedral invariance
        # Sample generations and count how many DON'T break the verifier
        dihedral_failures = 0
        for _ in tqdm.trange(INVARIANCE_ATTEMPTS, desc=f'{task_id}: dihedral attempts', leave=False):
            try:
                generate_examples(task_id, num_examples=num_train + num_test, dihedral_transform=True)
                # Success means we found a transform that breaks verifier
            except TransformError:
                # Couldn't find a transform that breaks verifier -> transform preserves it
                dihedral_failures += 1

        # Invariance = proportion of transforms that DON'T break the verifier
        dihedral_invariance = dihedral_failures / INVARIANCE_ATTEMPTS

        metadata: TaskMetadata = {
            'num_train': num_train,
            'num_test': num_test,
            'num_steps': num_steps,
            'color_invariance': round(color_invariance, 2),
            'dihedral_invariance': round(dihedral_invariance, 2),
        }

        yield task_id, metadata

def find_duplicate_verifiers() -> list[list[str]]:
    """
    Find verifiers with duplicate implementations.

    Returns:
        List of groups where each group contains task_ids with identical verifier code.
        Only groups with 2+ tasks are included.
    """
    verifiers_mapper = get_verifiers()
    verifier_codes: dict[str, str] = {}

    # Collect all verifier source codes
    for task_id, verifier in verifiers_mapper.items():
        # Get source code and normalize it (remove function signature)
        source = inspect.getsource(verifier)
        # Extract just the function body (everything after first line)
        lines = source.split('\n')
        body = '\n'.join(lines[1:])  # Skip function def line

        verifier_codes[task_id] = body

    # Group by identical code
    code_to_tasks: dict[str, list[str]] = {}
    for task_id, code in verifier_codes.items():
        if code not in code_to_tasks:
            code_to_tasks[code] = []
        code_to_tasks[code].append(task_id)

    # Return only groups with duplicates (2+ tasks)
    duplicate_groups = [tasks for tasks in code_to_tasks.values() if len(tasks) > 1]

    return duplicate_groups


def generate_metadata(
    output_path: str = '.',
    duplicate_verifiers_file: str = 'duplicate_verifiers.json',
    task_metadata_file: str = 'task_metadata.csv'
) -> None:
    """
    Generate and save metadata about verifiers and tasks.

    Writes two files:
    1. duplicate_verifiers.json - groups of task_ids with identical verifier implementations
    2. task_metadata.csv - metadata for each task including train/test counts, num_steps,
       color_invariance, and dihedral_invariance (float 0.0-1.0)

    Args:
        output_path: Directory to save metadata files
        duplicate_verifiers_file: Filename for duplicate verifiers output
        task_metadata_file: Filename for task metadata output
    """
    # Find and write duplicate verifiers
    print('Finding duplicate verifiers...')
    duplicate_groups = find_duplicate_verifiers()
    duplicates_path = os.path.join(output_path, duplicate_verifiers_file)
    with open(duplicates_path, 'w') as f:
        json.dump(duplicate_groups, f, indent=2)
    print(f'Wrote duplicate verifiers to {duplicates_path}')

    # Generate and write task metadata with tqdm progress bar
    verifiers_mapper = get_verifiers()
    num_tasks = len(verifiers_mapper)

    all_metadata = {}
    for task_id, metadata in tqdm.tqdm(generate_task_metadata(), total=num_tasks, desc='Generating task metadata'):
        all_metadata[task_id] = metadata

    metadata_path = os.path.join(output_path, task_metadata_file)
    with open(metadata_path, 'w', newline='') as f:
        # Define CSV headers
        fieldnames = ['task_id', 'num_train', 'num_test', 'num_steps', 'color_invariance', 'dihedral_invariance']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for task_id, metadata in all_metadata.items():
            row = {'task_id': task_id, **metadata}
            writer.writerow(row)
    print(f'Wrote task metadata to {metadata_path}')

if __name__ == '__main__':
    generate_metadata()