# -*- coding: utf-8 -*-

import SocketServer
#print SocketServer.ThreadingMixIn.daemon_threads

SocketServer.ThreadingMixIn.daemon_threads = True
import sampy
#print sampy.ThreadingXMLRPCServer.daemon_threads 
sampy.ThreadingXMLRPCServer.daemon_threads = True



from gavi.samp import Samp


samp = Samp()
#print samp.client
#print samp.client.client
#print samp.client.client.client
#print samp.client.client.client
#samp.client.enotifyAll("samp.app.echo", txt = "Hello world!")
#import pdb
#samp.client.client.client.process_request_thread.setDaemon(True)
#samp.client.client.client.process_request_thread.join()

#pdb.set_trace()
#samp.connect()

thread = samp.client.client._thread

try:
	while True:
		thread.join(100)
		if not thread.isAlive():
			break
except KeyboardInterrupt, e:
	print "error", e
samp.client.disconnect()
#samp.client.client._thread.join()