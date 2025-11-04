from sql_agent import prompt_template_baseline, sql_agent_rollout
from data_utils import load_spider_dataset
from agentlightning.types import Dataset
from agentlightning.algorithm.apo import APO
from agentlightning.adapter import TraceToMessages
from agentlightning import Trainer, configure_logger
import logging
import multiprocessing
from typing import Any, Dict, Tuple, cast

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def load_train_val_dataset() -> Tuple[Dataset[Dict[str, Any]], Dataset[Dict[str, Any]]]:
    """Load and split Spider dataset following APO pattern."""

    dataset_full = load_spider_dataset(
        "data/dev.json", database_dir="databases")

    dataset_full = dataset_full[:100]  # Use only 100 examples total

    # Split into train/val
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i]
                     for i in range(train_split)]  # 50 examples
    dataset_val = [dataset_full[i]
                   # 50 examples
                   for i in range(train_split, len(dataset_full))]

    return cast(Dataset[Dict[str, Any]], dataset_train), cast(Dataset[Dict[str, Any]], dataset_val)


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """Dump a copy of all logs produced by APO to a file."""
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def main() -> None:
    """Main training function following room_selector_apo.py pattern."""
    configure_logger()
    setup_apo_logger()

    openai_client = AsyncOpenAI()

    # Initialize APO algorithm
    algo = APO[Dict[str, Any]](
        openai_client,
        val_batch_size=10,
        gradient_batch_size=4,
        beam_width=2,
        branch_factor=2,
        beam_rounds=2,
    )

    trainer = Trainer(
        algorithm=algo,
        n_runners=8,
        initial_resources={
            "prompt_template": prompt_template_baseline()
        },
        adapter=TraceToMessages(),
    )

    dataset_train, dataset_val = load_train_val_dataset()

    trainer.fit(agent=sql_agent_rollout,
                train_dataset=dataset_train, val_dataset=dataset_val)


if __name__ == "__main__":
    main()
