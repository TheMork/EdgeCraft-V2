import re

with open("src/optimization/sweep_engine.py", "r") as f:
    content = f.read()

# 1. Add os import and os.makedirs to _init_db
content = re.sub(
    r'class SweepEngine:\n    def __init__\(self, db_path: str = "results/sweeps.db"\):\n        self.db_path = db_path\n        self._init_db\(\)\n\n    def _init_db\(self\):',
    r'class SweepEngine:\n    def __init__(self, db_path: str = "results/sweeps.db"):\n        self.db_path = db_path\n        self._init_db()\n\n    def _init_db(self):\n        import os\n        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)',
    content
)

# 2. Add cancellation check to run_grid_search
grid_search_replacement = r'''
            with multiprocessing.Pool(processes=processes) as pool:
                for result in pool.imap_unordered(SweepEngine._run_task, items):
                    # Check for cancellation
                    if completed % max(1, len(param_list) // 100) == 0:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('SELECT status FROM sweep_jobs WHERE job_id = ?', (job_id,))
                            row = cursor.fetchone()
                            if row and row[0] == "cancelled":
                                pool.terminate()
                                return

                    completed += 1
                    self.save_result(job_id, result['parameters'], result['metrics'])
                    if completed % 10 == 0 or completed == len(param_list):
                        self.update_job_status(job_id, "running", progress=completed)
'''
content = re.sub(
    r'\s*with multiprocessing.Pool\(processes=processes\) as pool:\n\s*for result in pool.imap_unordered\(SweepEngine._run_task, items\):\n\s*completed \+= 1\n\s*self.save_result\(job_id, result\[\'parameters\'\], result\[\'metrics\'\]\)\n\s*if completed % 10 == 0 or completed == len\(param_list\):\n\s*self.update_job_status\(job_id, "running", progress=completed\)',
    grid_search_replacement,
    content
)

# 3. Add cancellation check to run_bayesian_optimization
bayesian_opt_replacement = r'''
            runner = SimulationRunner(
                strategy,
                config.symbol,
                config.start_date,
                config.end_date,
                db_host=db_host,
                initial_balance=config.initial_balance,
                leverage=config.leverage
            )

            # Check for cancellation
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT status FROM sweep_jobs WHERE job_id = ?', (job_id,))
                row = cursor.fetchone()
                if row and row[0] == "cancelled":
                    trial.study.stop()
                    return -float('inf')

            result = runner.run()
'''
content = re.sub(
    r'\s*runner = SimulationRunner\(\n\s*strategy,\n\s*config.symbol,\n\s*config.start_date,\n\s*config.end_date,\n\s*db_host=db_host,\n\s*initial_balance=config.initial_balance,\n\s*leverage=config.leverage\n\s*\)\n\s*result = runner.run\(\)',
    bayesian_opt_replacement,
    content
)

with open("src/optimization/sweep_engine.py", "w") as f:
    f.write(content)
