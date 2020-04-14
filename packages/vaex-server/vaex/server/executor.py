import vaex.events
import vaex.execution


class Executor(vaex.execution.Executor):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.remote_calls = 0  # how many times did we call the server

    def _rmi(self, df, methodname, args, kwargs):
        # TODO: turn evaluate into a task
        return self.client._rmi(df, methodname, args, kwargs)

    def execute(self):
        try:
            self.signal_begin.emit()
            cancelled = False
            while not cancelled:
                tasks_df = self.local.tasks = self._pop_tasks()
                if not tasks_df:
                    break
                df = tasks_df[0].df
                tasks_df[0].signal_progress.connect(self.signal_progress.emit)
                for task in tasks_df:
                    task.signal_progress.emit(0)
                try:
                    results = self.client.execute(df, tasks_df)
                except vaex.execution.UserAbort:
                    self.signal_cancel.emit()
                    raise
                self.remote_calls += 1
                for task, result in zip(tasks_df, results):
                    if task.cancelled:
                        self.signal_cancel.emit()
                        task.reject(vaex.execution.UserAbort("cancelled"))
                    else:
                        task._result = result
                        task.fulfill(result)
            self.signal_end.emit()
        finally:
            self.tasks = []
