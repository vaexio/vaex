import vaex.events


class Executor:
    def __init__(self, client):
        self.client = client
        self.tasks = []
        self.signal_begin = vaex.events.Signal("begin")
        self.signal_progress = vaex.events.Signal("progress")
        self.signal_end = vaex.events.Signal("end")
        self.signal_cancel = vaex.events.Signal("cancel")
        self.remote_calls = 0  # how many times did we call the server

    def schedule(self, task):
        self.tasks.append(task)

    def _rmi(self, df, methodname, args, kwargs):
        # TODO: turn evaluate into a task
        return self.client._rmi(df, methodname, args, kwargs)

    def execute(self):
        tasks = list(self.tasks)
        try:
            self.signal_begin.emit()
            dfs = set(task.df for task in tasks)
            for df in dfs:
                tasks_df = [task for task in tasks if task.df is df]
                # chain it to the first task
                tasks_df[0].signal_progress.connect(self.signal_progress.emit)
                for task in tasks_df:
                    task.signal_progress.emit(0)
                results = self.client.execute(df, tasks_df)
                self.remote_calls += 1
                for task, result in zip(tasks_df, results):
                    if task.cancelled:
                        task.reject(vaex.execution.UserAbort("cancelled"))
                    else:
                        task._result = result
                        task.fulfill(result)
            self.signal_end.emit()
        finally:
            self.tasks = []
