__author__ = 'maartenbreddels'

import time

import vaex.ui.plugin
from vaex.ui.qt import *


#import vaex.plot_windows

import logging
#import vaex.ui.undo as undo

logger = logging.getLogger("vaex.ui.plugin.animation")

class AnimationPlugin(vaex.ui.plugin.PluginLayer):
	name = "animation"
	def __init__(self, parent, layer):
		super(AnimationPlugin, self).__init__(parent, layer)
		layer.plug_page(self.plug_page, "Animation", 6.0, 1.0)

		self.timer_sequence = QtCore.QTimer(parent)
		self.timer_sequence.timeout.connect(self.on_timeout_sequence)
		self.timer_sequence.setInterval(40) # max 50 fps to save cpu/battery?
		self.has_snapshots = self.dataset.has_snapshots()

		self.no_snapshots = self.dataset.rank1s[list(self.dataset.rank1s.keys())[0]].shape[0] if self.has_snapshots else 0
		#self.plot_original = self.dialog.plot
		#self.dialog.plot = self.plot_wrapper


		def on_plot_finished(plot_window, figure):
			#if self.timer_sequence.isActive():
			#	self.dialog.update_grids()
			#else:
			if self.record_frames:
				index = self.dataset.selected_serie_index
				path = self.frame_template.format(index=index)
				figure.savefig(path)
				if index == self.no_snapshots-1:
					self.record_frames = False
					self.frame_template = None

		self.layer.plot_window.signal_plot_finished.connect(on_plot_finished)

		self.record_frames = False
		self.frame_template = None
		#self.dataset.serie_index_selection_listeners.append(self.on_snapshot_select)
		# instead of binding to the event listener, wrap it so we are sure we execute it afterwards
		# this assumes the function isn't bound yet
		#def wrapper(index, previous=self.dialog.onSerieIndexSelect):
		#	previous(index)
		#	print ">>>>3", index, self.dataset.selected_serie_index
		#	self.box_sequence.setCurrentIndex(self.dataset.selected_serie_index)
		#self.dialog.onSerieIndexSelect = wrapper
		def on_sequence_index_change(dataset, index):
			self.box_sequence.setCurrentIndex(index)
		self.dataset.signal_sequence_index_change.connect(on_sequence_index_change)
		#print "set time to zero" * 100
		self.dataset.variables["time"] = str(0.)


		self.timer_realtime = QtCore.QTimer(parent)
		self.timer_realtime.timeout.connect(self.on_timeout_realtime)
		self.timer_realtime.setInterval(40) # max 50 fps to save cpu/battery?
		self.time_start = time.time()

	def clean_up(self):
		self.timer_sequence.stop()
		self.timer_realtime.stop()
		#self.dataset.serie_index_selection_listeners.remove(self.on_snapshot_select)

	def on_timeout_realtime(self):
		self.dataset.variables["time"] = str(self.time_start - time.time())
		#self.dialog.compute()
		#self.layer.jobs_manager.execute()


	def on_timeout_sequence(self):
		#self.dialog.update_grids()
		#if self.
		index = self.dataset.selected_serie_index + 1
		if index < self.no_snapshots:
			msg = "snapshot %d/%d" % (index+1, self.no_snapshots)
			self.layer.plot_window.message(msg, index=-1)
			self.dataset.selectSerieIndex(index)
			for layer in self.layer.plot_window.layers:
				if layer.dataset != self.dataset:
					layer.dataset.selectSerieIndex(index)
			#$self.layer.
			self.layer.jobs_manager.execute()
		else:
			self.layer.plot_window.message(None, index=-1)
			self.timer_sequence.stop()
			self.stop_animation()


	def plug_page(self, page):
		layout = self.layout = QtGui.QGridLayout()
		page.setLayout(self.layout)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		layout.setAlignment(QtCore.Qt.AlignTop)

		row = 0

		has_snapshots = self.dataset.has_snapshots()



		self.group_box_sequence = QtGui.QGroupBox("Snapshot animation", page)
		layout.addWidget(self.group_box_sequence, row, 1)
		row += 1
		layout_sequence = self.layout_sequence = QtGui.QGridLayout()
		#self.group_box_sequence.setLayout(self.layout_sequence)
		layout_sequence.setSpacing(0)
		layout_sequence.setContentsMargins(0,0,0,0)
		layout_sequence.setAlignment(QtCore.Qt.AlignTop)
		row_sequence = 0


		self.layout_control = QtGui.QHBoxLayout()
		self.layout_control.setSpacing(0)
		logger.debug("3")

		def add_control_button(text, handler):
			button = QtGui.QToolButton(self.group_box_sequence)
			button.setText(text)
			#button.setAutoDefault(False)
			button.setContentsMargins(0, 0, 0, 0)
			button.setEnabled(has_snapshots)
			#button.setFlat(True)
			self.layout_control.addWidget(button, 1)
			def wrap_handler():
				handler()
				self.layer.jobs_manager.execute()

			button.clicked.connect(wrap_handler)
			
		def select(index):
			self.dataset.selectSerieIndex(index)
			for layer in self.layer.plot_window.layers:
				if layer.dataset != self.dataset:
					layer.dataset.selectSerieIndex(index)
			#$self.layer.
			self.layer.jobs_manager.execute()
		def do_begin():
			select(0)
		def do_end():
			select(self.no_snapshots-1)
		def do_prev():
			select(max(0, self.dataset.selected_serie_index-1))
		def do_next():
			select(min(self.no_snapshots-1, self.dataset.selected_serie_index+1))
		def do_prev_ten():
			select(max(0, self.dataset.selected_serie_index-10))
		def do_next_ten():
			select(min(self.no_snapshots-1, self.dataset.selected_serie_index+10))
		add_control_button("|<", do_begin)
		add_control_button("<<", do_prev_ten)
		add_control_button("<", do_prev)
		add_control_button(">", do_next)
		add_control_button(">>", do_next_ten)
		add_control_button(">|", do_end)

		self.layout_sequence.addLayout(self.layout_control, row_sequence, 1)
		row_sequence += 1
		#self.label_sequence_info = QtGui.QLabel("""
#A snapshot animation is an animation where
#each frame shows a different snapshot,
#where usually the next snapshot is a next moment in time).
#		""".strip(), self.group_box_sequence)
		#self.layout_sequence.addWidget(self.label_sequence_info, row_sequence, 1)
		row_sequence += 1

		self.box_sequence = QtGui.QComboBox(self.group_box_sequence)
		self.box_sequence.addItems([str(k) for k in range(self.no_snapshots)])
		self.box_sequence.setCurrentIndex(self.dataset.current_sequence_index())
		self.box_sequence.setEnabled(has_snapshots)
		def on_sequence_change(index):
			if index != self.dataset.current_sequence_index():
				self.dataset.selectSerieIndex(index)
				for layer in self.layer.plot_window.layers:
					if layer.dataset != self.dataset:
						layer.dataset.selectSerieIndex(index)
				#self.dataset.selectSerieIndex(index)
				self.layer.jobs_manager.execute()
		self.box_sequence.currentIndexChanged.connect(on_sequence_change)
		self.layout_sequence.addWidget(self.box_sequence, row_sequence, 1)
		row_sequence += 1

		self.button_play_sequence = QtGui.QPushButton("Play(snapshots)", self.group_box_sequence)
		self.button_play_sequence.setCheckable(True)
		self.button_play_sequence.setAutoDefault(False)
		self.button_play_sequence.setEnabled(has_snapshots)
		if has_snapshots:
			self.button_play_sequence.setToolTip("This data set has no snapshots")
		layout_sequence.addWidget(self.button_play_sequence, row_sequence, 1)
		row_sequence += 1

		def on_toggle_play_sequence(checked):
			if checked:
				self.start_animation(self.button_play_sequence)
				#self.dataset.selectSerieIndex(0)
				self.layer.jobs_manager.execute()
				self.timer_sequence.start()
			else:
				self.timer_sequence.stop()
				self.stop_animation()
		self.button_play_sequence.toggled.connect(on_toggle_play_sequence)


		self.button_record_sequence = QtGui.QPushButton("Record(snapshots)", self.group_box_sequence)
		self.button_record_sequence.setCheckable(True)
		self.button_record_sequence.setAutoDefault(False)
		has_snapshots = self.dataset.has_snapshots()
		self.button_record_sequence.setEnabled(has_snapshots)
		if has_snapshots:
			self.button_record_sequence.setToolTip("This data set has no snapshots")
		layout_sequence.addWidget(self.button_record_sequence, row_sequence, 1)
		row_sequence += 1

		def on_toggle_record_sequence(checked):
			self.record_frames = False
			if checked:
				directory = getdir(self.parent, "Choose where to save frames", "")
				if directory:
					self.frame_template = os.path.join(directory, "%s_{index:05}.png" % self.dataset.name)
					self.frame_template = gettext(self.parent, "template for frame filenames", "template:", self.frame_template)
					if self.frame_template:
						self.record_frames = True
						#self.dataset.selectSerieIndex(0)
						self.layer.jobs_manager.execute()
						self.start_animation(self.button_record_sequence)
						self.timer_sequence.start()
			else:
				self.timer_sequence.stop()
				self.stop_animation()
		self.button_record_sequence.toggled.connect(on_toggle_record_sequence)


		self.group_box_realtime = QtGui.QGroupBox("Realtime animation", page)
		layout.addWidget(self.group_box_realtime, row, 1)
		layout_realtime = self.layout_realtime = QtGui.QGridLayout()
		self.group_box_realtime.setLayout(self.layout_realtime)
		layout_realtime.setSpacing(0)
		layout_realtime.setContentsMargins(0,0,0,0)
		layout_realtime.setAlignment(QtCore.Qt.AlignTop)
		row_sequence = 0
		self.button_play_realtime = QtGui.QPushButton("Play(time)", self.group_box_sequence)
		self.button_play_realtime.setCheckable(True)
		self.button_play_realtime.setAutoDefault(False)
		if has_snapshots:
			self.button_play_realtime.setToolTip("This data set has no snapshots")
		layout_realtime.addWidget(self.button_play_realtime, row_sequence, 1)
		row_sequence += 1

		def on_toggle_play_realtime(checked):
			if checked:
				self.start_animation(self.button_play_realtime)
				self.time_start = time.time()
				self.timer_realtime.start()
			else:
				self.timer_realtime.stop()
				self.stop_animation()
			self.button_play_realtime.toggled.connect(on_toggle_play_realtime)



	def start_animation(self, source_widget):
		widgets = [self.button_play_sequence, self.button_record_sequence, self.box_sequence]
		for widget in widgets:
			if source_widget != widget:
				widget.setEnabled(False)

	def stop_animation(self):
		widgets = [self.button_play_sequence, self.button_record_sequence, self.box_sequence]
		for widget in widgets:
			widget.setEnabled(True)

		self.button_play_sequence.setChecked(False)
		self.button_record_sequence.setChecked(False)

