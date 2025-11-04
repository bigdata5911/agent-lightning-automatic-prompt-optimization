import os
from spider_eval.exec_eval import eval_exec_match


def evaluate_query(
    query: str,
    ground_truth: str,
    database: str,
    raise_on_error: bool = True
) -> float:
    """Evaluate SQL query using Spider's execution-based evaluation.

    This follows the exact pattern from agent-lightning/examples/spider/sql_agent.py
    """
    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(
                f"Database file {database} does not exist.")

        # Parameters following the default setting from agent-lightning
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )

        if exec_score == 1:
            return 1.0
        else:
            return 0.0

    except Exception as e:
        if raise_on_error:
            raise
        else:
            print(f"Error evaluating query: {e}")
            return 0.0
