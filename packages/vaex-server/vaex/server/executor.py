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
        self.run(self.execute_async())

    async def execute_async(self):
        self.signal_begin.emit()
        cancelled = False
        # TODO: we only support UserAbort, not task._toreject
        while not cancelled:
            tasks_df = self.local.tasks = self._pop_tasks()
            if not tasks_df:
                break
            df = tasks_df[0].df
            for task in tasks_df:
                # we hook on the global progress indicator to all tasks
                # this way cancelling via the global progress event will cancel
                # all tasks
                task.signal_progress.connect(lambda x: all(self.signal_progress.emit(x)))
                task.signal_progress.emit(0)
            try:
                results = await self.client.execute_async(df, tasks_df)
            except vaex.execution.UserAbort:
                self.signal_cancel.emit()
                raise
            self.remote_calls += 1
            for task, result in zip(tasks_df, results):
                if task.cancelled:
                    cancelled = True
                    self.signal_cancel.emit()
                    task.reject(vaex.execution.UserAbort("cancelled"))
                else:
                    task._result = result
                    task.fulfill(result)
        self.signal_end.emit()
