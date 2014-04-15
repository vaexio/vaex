# -*- coding: utf-8 -*-
#from sampy import *
#from SocketServer import ThreadingMixIn
import sampy
from gavi import logging as logging_


logger = logging_.getLogger("gavi.samp")


class Samp(object):
	def __init__(self):
		self.client = sampy.SAMPIntegratedClient(metadata = {"samp.name":"Client 1",
										"samp.description.text":"Test Client 1",
										"gavi.samp.version":"0.01"}, callable=True)
		connected = False
		try:
			self.client.connect()
			connected = True
		except sampy.SAMPHubError, e:
			print "error connecting to hub", e
		
		if connected:
			#self.client.client._thread.setDaemon(False)
			logger.info("connected to SAMP hub")
			logger.info("binding events")
			self.client.bindReceiveCall			("table.load.votable", self._onSampCall)
			self.client.bindReceiveNotification	("table.load.votable", self._onSampNotification)
			self.client.bindReceiveNotification	("table.highlight.row", self._onSampNotification)
			#self.client.bindReceiveMessage("table.load.votable", self._onSampCall)
			#self.client.bindReceiveResponse("table.load.votable", self._onSampCall)
			
			#self.client.bindReceiveCall("samp.*", self._onSampCall)
			#self.client.bindReceiveNotification("samp.*", self._onSampNotification)
			
			#self.client.bindReceiveCall("table.*", self._onSampCall)
			#self.client.bindReceiveNotification("table.*", self._onSampNotification)
			#self.client.bindReceiveMessage("table.*", self._onSampCall)
			#self.client.bindReceiveResponse("table.*", self._onSampCall)
			
			#self.client.bindReceiveMessage("table.votable.*", self._onSampCall)
			#self.client.bindReceiveResponse("table.votable.*", self._onSampCall)
			
	#def connect(self):
	#	self.client.connect()
		
	def _onSampNotification(self, private_key, sender_id, mtype, params, extra):
		print "Notification:", `private_key`, `sender_id`, `mtype`, `params`, `extra`
		
	def _onSampCall(self, private_key, sender_id, msg_id, mtype, params, extra):
		print "----"
		try:
			print "Call:", `private_key`, `sender_id`, `msg_id`, `mtype`, `params`, `extra`
			self.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result = {"txt": "printed"})
		except:
			print "errrrrrrororrrr hans!"

