from __future__ import print_function
__author__ = 'maartenbreddels'
import aplus

def function_with_error(value):
	print("received: %r" % value)
	raise Exception
def function_prints(value):
	print("received: %r" % value)
	return value

def error_handler(value):
	print("error: %r" % value)


p1 = aplus.Promise()
p2 = p1.then(function_with_error, error_handler)
p2.then(function_with_error, None).end()
p1.fulfill(1)
print("done")