from typing import TypedDict
import random

from dsl import Grid, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror
from utils import is_grid

import generators
import verifiers

class GenerationError(Exception):
    """Failed to generate examples"""
    pass

class TransformError(Exception):
    """Failed to find a transform that breaks verification"""
    pass

class ExamplePair(TypedDict):
    input: Grid
    output: Grid

class TaskMetadata(TypedDict):
    # task metadata
    num_train: int
    num_test: int
    # verifier metadata
    color_invariance: float
    dihedral_invariance: float
    num_steps: int

GEN_ATTEMPTS = 100
COLOR_ATTEMPTS = 100
INVARIANCE_THRESHOLD = 0.5
def generate_examples(
    task_id: str,
    num_examples: int = 4,
    color_transform: bool = False,
    dihedral_transform: bool = False,
    difficulty_range: tuple[float, float] = (0, 1),
) -> list[ExamplePair]:
    generator = getattr(generators, 'generate_' + task_id)
    verifier = getattr(verifiers, 'verify_' + task_id)
    seen = set()
    examples: list[ExamplePair] = []
    for _ in range(num_examples):
        success = False

        for _ in range(GEN_ATTEMPTS):
            try:
                example: ExamplePair = generator(difficulty_range[0], difficulty_range[1])
            except:
                continue

            if not is_grid(example['input']) or not is_grid(example['output']):
                continue

            if example['input'] == example['output']:
                continue

            input_hash = hash(example['input'])
            if input_hash in seen:
                continue
            seen.add(input_hash)

            try:
                if verifier(example['input']) != example['output']:
                    continue
            except Exception:
                # Verifier crashed on this example - skip it
                continue

            examples.append(example)
            success = True
            break

        if not success:
            raise GenerationError(f"Failed to generate example for {task_id}")

    def breaks_verification(transformed: list[ExamplePair]) -> bool:
        """Check if transformation breaks verification (we want it to fail)"""
        for ex in transformed:
            try:
                if verifier(ex['input']) != ex['output']:
                    return True
            except Exception:
                # Verifier crashed - transformation broke it
                return True
        return False

    def apply_transform_until_breaks(transform_generator, error_msg: str) -> list[ExamplePair]:
        """Try transformations until one breaks verification"""
        for transformed in transform_generator:
            if breaks_verification(transformed):
                return transformed
        raise TransformError(error_msg)

    if color_transform:
        # Get all unique colors from examples
        all_colors = {
            cell
            for example in examples
            for grid in (example['input'], example['output'])
            for row in grid
            for cell in row
        }

        colors = list(all_colors)

        def color_transform_generator():
            for _ in range(COLOR_ATTEMPTS):
                # Create random new palette (not the same as original)
                new_colors = [random.randint(0, 9) for _ in colors]

                # Ensure it's not the identity mapping
                if new_colors == colors:
                    continue

                # Create color mapping
                color_map = dict(zip(colors, new_colors))

                # Apply permutation to grid
                def permute_grid(grid: Grid) -> Grid:
                    return tuple(tuple(color_map[cell] for cell in row) for row in grid)

                yield [
                    ExamplePair(input=permute_grid(ex['input']), output=permute_grid(ex['output']))
                    for ex in examples
                ]

        examples = apply_transform_until_breaks(
            color_transform_generator(),
            "Failed to find color transformation that breaks verification"
        )

    if dihedral_transform:
        # 7 dihedral transformations (excluding identity)
        dihedral_ops = [rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror]
        random.shuffle(dihedral_ops)

        def dihedral_transform_generator():
            for transform in dihedral_ops:
                yield [
                    ExamplePair(input=transform(ex['input']), output=transform(ex['output']))
                    for ex in examples
                ]

        examples = apply_transform_until_breaks(
            dihedral_transform_generator(),
            "Failed to find dihedral transformation that breaks verification"
        )

    return examples

MIN_STEPS = 12
def choose_eval_tasks(metadata: dict[str, TaskMetadata], seed: int):
    """
    Filter tasks for evaluation based on:
    1. Tasks unsolved by icecuber (score == -1)
    2. Tasks with num_steps >= MIN_STEPS
    3. Choose one random representative from each duplicate group
    """
    import csv
    import json

    random.seed(seed)

    # Load icecuber results
    icecuber_unsolved = set()
    with open('icecuber.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['score'] == '-1':
                icecuber_unsolved.add(row['taskid'])

    # Load duplicate verifiers
    with open('duplicate_verifiers.json', 'r') as f:
        duplicate_groups = json.load(f)

    # Create mapping of task_id -> representative task_id
    task_to_representative = {}
    for group in duplicate_groups:
        # Choose one random representative from each group
        representative = random.choice(group)
        for task_id in group:
            task_to_representative[task_id] = representative

    # Filter metadata
    filtered = {}
    for task_id, task_metadata in metadata.items():
        # Filter by icecuber unsolved
        if task_id not in icecuber_unsolved:
            continue

        if task_metadata['num_steps'] < MIN_STEPS:
            continue

        # Choose one from duplicates
        representative = task_to_representative.get(task_id, task_id)
        if representative != task_id:
            continue

        filtered[task_id] = task_metadata

    return filtered

def load_task_metadata() -> dict[str, TaskMetadata]:
    """Load task metadata from CSV file."""
    import csv

    metadata = {}
    with open('task_metadata.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row['task_id']
            metadata[task_id] = TaskMetadata(
                num_train=int(row['num_train']),
                num_test=int(row['num_test']),
                num_steps=int(row['num_steps']),
                color_invariance=float(row['color_invariance']),
                dihedral_invariance=float(row['dihedral_invariance']),
            )
    return metadata


def generate_dataset(
    task_metadata: dict[str, TaskMetadata],
    seed: int
):
    """
    Generate dataset for tasks.

    Args:
        task_metadata: Dictionary mapping task_id to TaskMetadata
        seed: Random seed for reproducibility

    Yields:
        tuple of (task_id, {'train': list[ExamplePair], 'test': list[ExamplePair]})
    """

    # Generate examples for each task
    for task_id, metadata in task_metadata.items():
        # Set seed for this task (deterministic but different per task)
        # assuming task_id is 32bit hex num
        random.seed(seed ^ int(task_id, 16))

        # Generate all examples with appropriate transformations
        # Apply transforms if invariance < threshold (more dependent than invariant)
        num_examples = metadata['num_train'] + metadata['num_test']
        while True:
            try:
                examples = generate_examples(
                    task_id,
                    num_examples=num_examples,
                    color_transform=metadata['color_invariance'] < INVARIANCE_THRESHOLD,
                    dihedral_transform=metadata['dihedral_invariance'] < INVARIANCE_THRESHOLD
                )
                break
            except TransformError:
                continue

        # Split into train and test
        train_examples = examples[:metadata['num_train']]
        test_examples = examples[metadata['num_train']:]

        yield task_id, {
            'train': train_examples,
            'test': test_examples
        }

def print_dataset_tasks(
    task_metadata: dict[str, TaskMetadata],
    seed: int
):
    """
    Generate and print JSONs for each task individually.

    Args:
        task_metadata: Dictionary mapping task_id to TaskMetadata
        seed: Random seed for reproducibility
    """
    import json

    for task_id, task in generate_dataset(task_metadata, seed=seed):
        task_json = {task_id: task}
        print(json.dumps(task_json, separators=(',', ':')))

def save_dataset_to_file(
    task_metadata: dict[str, TaskMetadata],
    output_path: str,
    seed: int
):
    """
    Generate and save dataset to file with progress bar.

    Args:
        task_metadata: Dictionary mapping task_id to TaskMetadata
        output_path: Path to save the JSON file
        seed: Random seed for reproducibility
    """
    import json
    import tqdm

    total_tasks = len(task_metadata)

    # Generate dataset with progress bar
    dataset = {}
    pbar = tqdm.tqdm(
        generate_dataset(task_metadata, seed=seed),
        total=total_tasks
    )
    for task_id, task in pbar:
        pbar.set_description(f'Generating {task_id}')
        dataset[task_id] = task

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(dataset, f, separators=(',', ':'))

    print(f'Saved dataset with {len(dataset)} tasks to {output_path}')


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Generate ARC dataset')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, help='Output file path (for save mode)')

    args = parser.parse_args()

    # Load and filter metadata
    metadata = load_task_metadata()
    metadata = choose_eval_tasks(metadata, args.seed)

    # for task_id in metadata.keys():
    #     print(task_id)

    if args.output:
        solvable_count = sum(
            1 for m in metadata.values()
            if m['color_invariance'] >= INVARIANCE_THRESHOLD and m['dihedral_invariance'] >= INVARIANCE_THRESHOLD
        )
        print(f"Tasks solvable by verifiers: {solvable_count}/{len(metadata)} ({round(100 * solvable_count / len(metadata), 2)}%)")
        save_dataset_to_file(metadata, args.output, args.seed)
    else:
        print_dataset_tasks(metadata, args.seed)
