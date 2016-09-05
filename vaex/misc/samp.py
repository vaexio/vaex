# -*- coding: utf-8 -*-
#from sampy import *
#from SocketServer import ThreadingMixIn
import sampy
from gavi import logging as logging_
import threading

logger = logging_.getLogger("gavi.samp")


class Samp(object):
	def __init__(self, daemon=True, name=None):
		self.client = sampy.SAMPIntegratedClient(metadata = {"samp.name":"Gavi client" if name is None else name,
										"samp.description.text": "Gavi client" if name is None else name,
										"gavi.samp.version":"0.01"}, callable=True)


		# sampy doesn't make this thread Daeamon, so the python process never stops on the cmd line
		# this fixes that
		def _myrun_client():
			if self.client.client._callable:
				self.client.client._thread = threading.Thread(target = self.client.client._serve_forever)
				self.client.client._thread.setDaemon(True)
				self.client.client._thread.start()
		if daemon:
			self.client.client._run_client = _myrun_client
		connected = False
		try:
			self.client.connect()
			connected = True
		except sampy.SAMPHubError as e:
			print(("error connecting to hub", e))
		
		if connected:
			#self.client.client._thread.setDaemon(False)
			logger.info("connected to SAMP hub")
			logger.info("binding events")
			self.client.bindReceiveCall			("table.load.votable", self._onTableLoadVotable)
			self.client.bindReceiveNotification	("table.load.votable", self._onTableLoadVotable)
			#self.client.bindReceiveNotification	("table.highlight.row", self._onSampNotification)
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
		self.tableLoadCallbacks = []
	
	def _onTableLoadVotable(self, private_key, sender_id, msg_id, mtype, params, extra):
		print(("Msg:", repr(private_key), repr(sender_id), repr(msg_id), repr(mtype), repr(params), repr(extra)))
		try:
			url = params["url"]
			table_id = params["table-id"]
			name = params["name"]
			for callback in self.tableLoadCallbacks:
				callback(url, table_id, name)
		except:
			logger.exception("event handler failed")
		
		if msg_id != None: # if SAMP call, send a reply
			self.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result = {"txt": "loaded"})
		
	def _onSampNotification(self, private_key, sender_id, mtype, params, extra):
		print(("Notification:", repr(private_key), repr(sender_id), repr(mtype), repr(params), repr(extra)))
		
	def _onSampCall(self, private_key, sender_id, msg_id, mtype, params, extra):
		print("----")
		try:
			print(("Call:", repr(private_key), repr(sender_id), repr(msg_id), repr(mtype), repr(params), repr(extra)))
			self.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result = {"txt": "printed"})
		except:
			print("errrrrrrororrrr hans!")

