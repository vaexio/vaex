import asyncio
import concurrent.futures
import logging
import threading

import vaex


logger = logging.getLogger("vaex.server.service")


class Service:
    def __init__(self, df_map):
        self.df_map = df_map

    def stop(self):
        pass

    def __getitem__(self, item):
        return self.df_map[item]

    def list(self):
        """Return a dict with dataframe information"""
        return {name: {
                       'length_original': df.length_original(),
                       'column_names': df.get_column_names(strings=True),
                       'dtypes': {name: vaex.encoding.dtype_encoding.encode(None, df.data_type(name)) for name in df.get_column_names(strings=True)},
                       'state': df.state_get()
                    } for name, df in self.df_map.items()
                }

    def _rmi(self, df, methodname, args, kwargs):
        method = getattr(df, methodname)
        return method(*args, **kwargs)

    async def execute(self, df, tasks, progress=None):
        assert df.executor.tasks == []
        tasks = [df.executor.schedule(task) for task in tasks]
        await df.execute_async()
        return [task.get() for task in tasks]


class Proxy:
    def __init__(self, service):
        self.service = service

    def __getitem__(self, item):
        return self.service[item]

    def stop(self):
        return self.service.stop()

    def list(self):
        return self.service.list()

    def _rmi(self, df, methodname, args, kwargs):
        return self.service._rmi(df, methodname, args, kwargs)


class AsyncThreadedService(Proxy):
    def __init__(self, service, thread_count, threads_per_job):
        super().__init__(service)
        self.threads_per_job = threads_per_job
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(thread_count)
        self.thread_local = threading.local()
        self.thread_pools = []

    def stop(self):
        self.thread_pool.shutdown()
        for thread_pool in self.thread_pools:
            thread_pool.shutdown()

    async def execute(self, df, tasks, progress=None):
        def execute():
            if not hasattr(self.thread_local, "executor"):
                logger.debug("creating thread pool and executor")
                self.thread_local.thread_pool = vaex.multithreading.ThreadPoolIndex(max_workers=self.threads_per_job)
                self.thread_local.executor = vaex.execution.ExecutorLocal(thread_pool=self.thread_local.thread_pool)
                self.thread_local.ioloop = loop = asyncio.new_event_loop()
                self.thread_pools.append(self.thread_local.thread_pool)
            async def execute_async():
                executor = self.thread_local.executor
                try:
                    if progress:
                        executor.signal_progress.connect(progress)
                    df.executor = executor
                    return await self.service.execute(df, tasks)
                finally:
                    if progress:
                        executor.signal_progress.disconnect(progress)
            return self.thread_local.ioloop.run_until_complete(execute_async())

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, execute)
