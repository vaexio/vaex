import ipyvuetify as vue

import ipywidgets as widgets

import traitlets


class PlotTemplatePlotly(vue.VuetifyTemplate):
    show_output = traitlets.Bool(False).tag(sync=True)
    new_output = traitlets.Bool(False).tag(sync=True)
    title = traitlets.Unicode('Vaex').tag(sync=True)

    drawer = traitlets.Bool(True).tag(sync=True)
    clipped = traitlets.Bool(False).tag(sync=True)
    model = traitlets.Any(True).tag(sync=True)
    floating = traitlets.Bool(False).tag(sync=True)
    dark = traitlets.Bool(False).tag(sync=True)
    mini = traitlets.Bool(False).tag(sync=True)
    components = traitlets.Dict(None, allow_none=True).tag(sync=True, **widgets.widget.widget_serialization)
    items = traitlets.Any([]).tag(sync=True)
    type = traitlets.Unicode('temporary').tag(sync=True)
    items = traitlets.List(['red', 'green', 'purple']).tag(sync=True)
    button_text = traitlets.Unicode('menu').tag(sync=True)
    drawers = traitlets.Any(['Default (no property)', 'Permanent', 'Temporary']).tag(sync=True)
    template = traitlets.Unicode('''

<v-app>
    <v-navigation-drawer style="width: 310px"
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
      clipped
      right
      absolute
      overflow
    >
      <h3>Output</h3>
      <output-widget />
    </v-navigation-drawer>

    <v-app-bar :clipped-left="clipped" absolute dense>
      <v-app-bar-nav-icon
        v-if="type !== 'permanent'"
        @click.stop="model = !model"
      ></v-app-bar-nav-icon>
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


    </v-app-bar>
    <v-content style="margin-top: 50px; padding-top: 0px">
          <main-widget/>
    </v-content>
</v-app>
''').tag(sync=True)
