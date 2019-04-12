from __future__ import print_function
__author__ = 'maartenbreddels'
import numpy as np
import unittest
import vaex.webserver
import vaex as vx
import vaex.webserver
import vaex.execution
import time

vx.set_log_level_exception()
vx.set_log_level_debug()

class JobSleep(vaex.webserver.JobFlexible):
	def execute(self):
		time.sleep(self.cost*self.fraction*0.2)
		self.done()

class TestQueue(unittest.TestCase):
	def test_queue(self):
		global count
		job_queue = vaex.webserver.JobQueue()
		job_executor = vaex.webserver.JobExecutor(job_queue=job_queue)
		jobs = []
		for i in range(20):
			job = JobSleep(1)
			job_queue.add(job)
			jobs.append(job)
		count = 0
		def add_job(prev_job):
			global count
			count += 1
			repeat = 0

			if count >= 20:
				repeat = 2
			if count >= 50:
				repeat = 0

			if count >= 70:
				repeat = 3
			if count >= 100:
				repeat = 0
			#repeat = 1

			for i in range(repeat):
				job = JobSleep(1)
				job_queue.add(job)
				jobs.append(job)
		job_queue.signal_job_finished.connect(add_job)
		try:
			job_executor.empty_queue()
		except KeyboardInterrupt:
			pass
		for i, job in enumerate(jobs):
			print(i, job.index, "elapsed", job.time_elapsed, job.fraction)



if __name__ == '__main__':
    unittest.main()

