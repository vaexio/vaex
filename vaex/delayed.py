__author__ = 'maartenbreddels'
import aplus

def promisify(value):
	# TODO, support futures etc..
	if isinstance(value, aplus.Promise):
		return value
	if isinstance(value, (list, tuple)):
		return aplus.listPromise(*list([promisify(k) for k in value]))
	else:
		return aplus.Promise.fulfilled(value)
def delayed(f):
	def wrapped(*args, **kwargs):
		#print "calling", f, "with", kwargs
		#key_values = kwargs.items()
		key_promise = list([(key, promisify(value)) for key, value in kwargs.items()])
		#key_promise = [(key, promisify(value)) for key, value in key_values]
		arg_promises = list([promisify(value) for value in args])
		kwarg_promises = list([promise for key, promise in key_promise])
		promises = arg_promises + kwarg_promises
		for promise in promises:
			def echo_error(exc, promise=promise):
				print("error with ", promise, "exception is", exc)
				#raise exc
			def echo(value, promise=promise):
				print("done with ", repr(promise), "value is", value)
			#promise.then(echo, echo_error)

		#print promises
		allarguments = aplus.listPromise(*promises)
		def call(_):
			kwargs_real = {key:promise.get() for key, promise in key_promise}
			args_real = list([promise.get() for promise in arg_promises])
			return f(*args_real, **kwargs_real)
		def error(exc):
			print("error", exc)
			raise exc
		return allarguments.then(call, error)
	return wrapped

@delayed
def delayed_args(*args):
	return args

@delayed
def delayed_list(l):
	return delayed_args(*l)