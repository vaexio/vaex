from __future__ import print_function
__author__ = 'breddels'
"""
Demonstrates combining Qt and tornado, both which want to have their own event loop.
The solution is to run tornado in a thread, the issue is that callbacks will then also be executed in this thread, and Qt doesn't like that.
To fix this, I show how to use execute the callback in the main thread, using a Qt signal/event in combination with Promises.

The output of the program is:
fetch page, we are in thread <_MainThread(MainThread, started 47200787479520)>
response is 191548 bytes, we are in thread <Thread(Thread-1, started daemon 47201018689280)>
the other thread should fulfil the result to this promise, we are in thread <Thread(Thread-1, started daemon 47201018689280)>
we received a promise, let us fulfill it, and are in thread <_MainThread(MainThread, started 47200787479520)>
let us set the background to black, we are in thread <_MainThread(MainThread, started 47200787479520)>

The magic happens in this line:
 .then(self.move_to_gui_thread)

Without it, you'll see something like this:
fetch page, we are in thread <_MainThread(MainThread, started 47822588292064)>
response is 191604 bytes, we are in thread <Thread(Thread-1, started daemon 47822819497728)>
let us set the background to black, we are in thread <Thread(Thread-1, started daemon 47822819497728)>
QPixmap: It is not safe to use pixmaps outside the GUI thread

"""
from aplus import Promise # https://github.com/xogeny/aplus
import threading
import tornado
from tornado.httpclient import AsyncHTTPClient
from PyQt4 import QtGui
from PyQt4 import QtCore
import sys


# tornado works with futures, this wraps it in a promise
def wrap_future_with_promise(future):
	promise = Promise()
	def callback(future):
		e = future.exception()
		if e:
			promise.reject(e)
		else:
			promise.fulfill(future.result())
	future.add_done_callback(callback)
	return promise


class Window(QtGui.QMainWindow):
	signal_promise = QtCore.pyqtSignal(object, object)

	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.button = QtGui.QPushButton("Async fetch using tornado", self)
		self.button.resize(self.button.sizeHint())
		self.button.clicked.connect(self.on_click)
		self.signal_promise.connect(self.on_signal_promise)

	def on_click(self, *args):
		print("fetch page, we are in thread", threading.currentThread())
		client = AsyncHTTPClient()
		future = client.fetch("http://www.google.com/")
		promise = wrap_future_with_promise(future)
		# without .then(self.move_to_gui_thread), Qt will complain
		promise.then(self.show_output)\
			.then(self.move_to_gui_thread)\
			.then(self.do_gui_stuff)\
			.then(None, self.on_error)

	def move_to_gui_thread(self, value):
		promise = Promise()
		print("the other thread should fulfil the result to this promise, we are in thread", threading.currentThread())
		self.signal_promise.emit(promise, value)
		return promise

	def on_signal_promise(self, promise, value):
		print("we received a promise, let us fulfill it, and are in thread", threading.currentThread())
		promise.fulfill(value)

	def on_error(self, error):
		print("error", error)

	def show_output(self, response):
		print("response is", len(response.body), "bytes, we are in thread", threading.currentThread())

	def do_gui_stuff(self, response):
		print("let us set the background to orange, we are in thread", threading.currentThread())
		# this Qt call should only be done from the main thread
		self.setStyleSheet("background-color: orange;")


# run the tornado loop in a seperate thread
thread = threading.Thread(target=lambda : tornado.ioloop.IOLoop.current().start())
thread.setDaemon(True)
thread.start()

app = QtGui.QApplication(sys.argv)
window = Window()
window.show()
app.exec_()
