import ipyvuetify as v
import ipywidgets as widgets
from traitlets import *


class PlotTemplate(v.VuetifyTemplate):
    show_output = Bool(False).tag(sync=True)
    new_output = Bool(False).tag(sync=True)
    title = Unicode('Vaex').tag(sync=True)

    drawer = Bool(True).tag(sync=True)
    clipped = Bool(False).tag(sync=True)
    model = Any(True).tag(sync=True)
    floating = Bool(False).tag(sync=True)
    dark = Bool(False).tag(sync=True)
    mini = Bool(False).tag(sync=True)
    components = Dict(None, allow_none=True).tag(sync=True, **widgets.widget.widget_serialization)
    items = Any([]).tag(sync=True)
    type = Unicode('temporary').tag(sync=True)
    items = List(['red', 'green', 'purple']).tag(sync=True)
    button_text = Unicode('menu').tag(sync=True)
    drawers = Any(['Default (no property)', 'Permanent', 'Temporary']).tag(sync=True)
    template = Unicode('''

<v-app>
    <v-navigation-drawer
      v-model="model"
      :permanent="type === 'permanent'"
      :temporary="type === 'temporary'"
      :clipped="clipped"
      :floating="floating"
      :mini-variant="mini"
      absolute
      overflow
    >

        <control-widget/>

      </v-list>

    </v-navigation-drawer>

    <v-navigation-drawer
      v-model="show_output"
      :temporary="type === 'temporary'"
      clipped="true"
      floating="false"
      right
      absolute
      overflow
    >
      <h3>Output</h3>
      <output-widget>
    </v-navigation-drawer>

    <v-toolbar :clipped-left="clipped" absolute dense>
      <v-toolbar-side-icon
        v-if="type !== 'permanent'"
        @click.stop="model = !model"
      ></v-toolbar-side-icon>
      <v-toolbar-title>{{title}} </v-toolbar-title>
      <v-spacer></v-spacer>


    <v-btn icon @click.stop="show_output = !show_output; new_output=false">
      <v-badge color="red" overlap>
        <template v-slot:badge v-if="new_output">
          <span>!</span>
        </template>
            <v-icon>error_outline</v-icon>
      </v-badge>
    </v-btn>


    </v-toolbar>
    <v-content>
          <main-widget/>
    </v-content>
<v-app>
''').tag(sync=True)