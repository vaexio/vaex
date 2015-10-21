from __future__ import print_function
import sys
import time

## @defgroup python Python
#@{
class ProgressBar(object):
	"""
		Implementation of progress bar on a console
		
		Usage::
		
			progressbar = ProgressBar(0, 100)
			progressbar.update(0)
			....
			progressbar.update(100)
		
		By default, the progress bar writes to stderr, so it doesn't clutter up log files when piping stdout 
	"""
	def __init__(self, min_value, max_value, format="%(bar)s: %(percentage) 6.2f%% %(timeinfo)s", width=40, barchar="#", emptychar="-", output=sys.stdout):
		"""		
			:param min_value: minimum value for update(..)
			:param format: format specifier for the output
			:param width: width of the progress bar's (excluding extra text)
			:param barchar: character used to print the bar
			:param output: where to write the output to
		"""
		self.min_value = min_value
		self.max_value = max_value
		self.format = format
		self.width = width
		self.barchar = barchar
		self.emptychar = emptychar
		self.output = output
		
		self.firsttime = True
		self.prevtime = time.time()
		self.starttime = self.prevtime
		self.prevfraction = 0
		self.firsttimedone = False
		
	## Updates the progress bar, writing a \\t char to stdout, and the progress bar
	# @param value the current progress of the bar, should be in the range min_value and max_value 
	def update(self, value):
		currenttime = time.time()
		if (self.max_value-self.min_value) == 0:
			fraction = 1.
		else:
			fraction = (float(value)-self.min_value)/(self.max_value-self.min_value)
		count = int(fraction * self.width + 0.5)
		space = self.width - count
		percentage = fraction * 100
		bar = "[" + (self.barchar * count) + (self.emptychar * space) + "]"
		if (fraction > 0) and ((fraction-self.prevfraction) > 0):
			if fraction == 1:
				elapsedtime = (currenttime-self.starttime)
				seconds = elapsedtime 
				minutes = seconds/60. 
				hours = minutes/60.
				timeinfo = "elapsed time  : % 8ds = % 4.1fm = % 2.1fh" % (seconds, minutes, hours)
			else:
				#estimatedtime = (currenttime-self.starttime)/(fraction) * (1-fraction)
				#estimatedtime = (currenttime-self.prevtime)/(fraction-self.prevfraction) * (1-fraction)
				estimatedtime = (currenttime-self.prevtime) / (fraction) * (1-fraction)
				seconds = estimatedtime 
				minutes = seconds/60.
				hours = minutes/60.
				timeinfo = "estimated time: % 8ds = % 4.1fm = % 2.1fh" % (seconds, minutes, hours)
		else:
			timeinfo = "estimated time: unknown                "
		s = self.format % {"bar":bar, "percentage":percentage, "timeinfo":timeinfo}
		if (fraction != 1) or (not self.firsttimedone): # last time print a newline char
			if not self.firsttime:
				self.firsttime = True
				print(s, end=' ', file=self.output)
			else:
				print("\r", end=' ', file=sys.stderr)
				print("\r" + s, end=' ', file=self.output)
			#print >>self.output, s,
		if (fraction == 1) and (not self.firsttimedone): # last time print a newline char
			self.firsttimedone = True
			print("", file=self.output)
			
		self.output.flush()
		#if fraction > 0:
		#	self.prevtime = currenttime
		#	self.prevfraction = fraction
			
			
def main():
	import time
	bar = ProgressBar(0, 100, width=20, barchar="*", emptychar=" ")
	for i in range(100+1):
		bar.update(i)
		time.sleep(0.05)
		
	bar = ProgressBar(0, 100, format="%(percentage) 6.2f%% %(timeinfo)s")
	for i in range(100+1):
		bar.update(i)
		time.sleep(0.05)
		
	
	bar = ProgressBar(0, 100)
	for i in range(100+1):
		bar.update(i)
		time.sleep(0.05)
		
	bar = ProgressBar(0, 100, format="%(bar)s: %(percentage) 6.2f%%")
	for i in range(100+1):
		bar.update(i)
		time.sleep(0.05)
		
if __name__ == "__main__":
	main()
		
#@}