from __future__ import absolute_import
import ipyvuetify as v
import ipywidgets as widgets
import traitlets
from traitlets import *
from . import traitlets as vt

def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()

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
    <v-content>
          <main-widget/>
    </v-content>
</v-app>
''').tag(sync=True)


class AnimatedCounter(v.VuetifyTemplate):
    characters = traitlets.List(traitlets.Unicode()).tag(sync=True)
    value = traitlets.Integer()
    format = traitlets.Unicode('{: 14,d}')
    prefix = traitlets.Unicode('').tag(sync=True)
    postfix = traitlets.Unicode('').tag(sync=True)

    @traitlets.observe('value')
    def _value(self, change):
        text = self.format.format(self.value)
        self.characters = [k.replace(' ', '&nbsp;') for k in text]

    template = traitlets.Unicode('''
          <div>
          {{ prefix }}
          <v-slide-y-transition :key=index v-for="(character, index) in characters" leave-absolute>
              <span :key="character" v-html='character'></span>
          </v-slide-y-transition>
          {{ postfix }}
          </div>
      ''').tag(sync=True)

class ProgressCircularNoAnimation(v.VuetifyTemplate):
    """v-progress-circular that avoids animations"""
    parts = traitlets.List(traitlets.Unicode()).tag(sync=True)
    width = traitlets.Integer().tag(sync=True)
    size = traitlets.Integer().tag(sync=True)
    value = traitlets.Float().tag(sync=True)
    color = traitlets.Unicode('{: 14,d}').tag(sync=True)
    text = traitlets.Unicode('{: 14,d}').tag(sync=True)
    hidden = traitlets.Bool(False).tag(sync=True)
    template = traitlets.Unicode('''
        <v-progress-circular v-if="!hidden" :key="value" :size="size" :width="width" :value="value" :color="color">{{ text }}</v-progress-circular>
        <v-progress-circular v-else style="visibility: hidden" :key="value" :size="size" :width="width" :value="value" :color="color">{{ text }}</v-progress-circular>
      ''').tag(sync=True)


class ExpressionTextArea(v.Textarea):
    df = traitlets.Any()
    valid = traitlets.Bool(True)

    @traitlets.default('v_model')
    def _v_model(self):
        columns = self.df.get_column_names(strings=False)
        if columns:
            if len(columns) >= 2:
                return columns[0] + " + " + columns[1]
            else:
                return columns[0]
        columns = self.df.get_column_names()
        return columns[0]

    @traitlets.default('label')
    def _label(self):
        return "Custom expression"

    @traitlets.default('placeholder')
    def _placeholder(self):
        return "Enter a custom expression"

    @traitlets.default('prepend_icon')
    def _prepend_icon(self):
        return 'functions'

    @traitlets.observe('v_model')
    def update_custom_selection(self, change):
        self.check_expression()

    def check_expression(self):
        try:
            self.df.validate_expression(self.v_model)
        except Exception as e:
            self.success_messages = None
            self.error_messages = str(e)
            self.valid = False
            return
        self.error_messages = None
        self.success_messages = "Looking good"
        self.valid = True
        return True


class ExpressionSelectionTextArea(ExpressionTextArea):
    # selection is v_model
    selection_name = traitlets.Any('default')

    @traitlets.default('v_model')
    def _v_model(self):
        columns = self.df.get_column_names(strings=False)
        return columns[0] + ' == 0'

    @traitlets.default('label')
    def _label(self):
        return "Filter by custom expression"

    @traitlets.default('placeholder')
    def _placeholder(self):
        return "Enter a custom (boolean) expression"

    @traitlets.default('prepend_icon')
    def _prepend_icon(self):
        return 'filter_list'

    @traitlets.observe('v_model')
    def update_custom_selection(self, change):
        if self.check_expression():
            self.update_selection()

    def update_selection(self):
        self.df.select(self.v_model, name=self.selection_name)

class ColumnPicker(v.VuetifyTemplate):
    df = traitlets.Any()
    items = traitlets.List(['foo', 'bar']).tag(sync=True)
    tooltip = traitlets.Unicode('Add example expression based on column...').tag(sync=True)
    template = traitlets.Unicode('''
        <v-layout>
            <v-menu offset-y>
                <template v-slot:activator="{ on: menu }">
                    <v-tooltip bottom>
                        {{ tooltip }}
                        <template v-slot:activator="{ on: tooltip}">
                            <v-btn
                                    v-on="{...menu, ...tooltip}" fab color='primary' small>
                                <v-icon>add</v-icon>
                            </v-btn>
                        </template>
                    </v-tooltip>
                </template>
                <v-list>
                    <v-list-item
                            v-for="(item, index) in items"
                            :key="index"
                            @click="menu_click(index)">
                        <v-list-item-title>{{ item }}</v-list-item-title>
                    </v-list-item>
                </v-list>
            </v-menu>
        </v-layout>''').tag(sync=True)

    @traitlets.default('items')
    def _items(self):
        return self.df.get_column_names()

    def vue_menu_click(self, data):
        pass


class ColumnExpressionAdder(ColumnPicker):
    component = traitlets.Any()
    target = traitlets.Unicode('v_model')

    def vue_menu_click(self, data):
        value = getattr(self.component, self.target)
        setattr(self.component, self.target, value + ' + ' + str(self.items[data]))


class ColumnSelectionAdder(ColumnPicker):
    component = traitlets.Any()
    target = traitlets.Unicode('v_model')

    def vue_menu_click(self, data):
        value = getattr(self.component, self.target)
        setattr(self.component, self.target, value + ' & ({} == 0)'.format(self.items[data]))


class SelectionEditor(v.VuetifyTemplate):
    df = traitlets.Any()
    input = traitlets.Any()
    adder = traitlets.Any()
    on_close = traitlets.Any()
    components = traitlets.Dict(None, allow_none=True).tag(sync=True, **widgets.widget.widget_serialization)

    @traitlets.default('components')
    def _components(self):
        return {'component-input': self.input, 'adder': self.adder}

    @traitlets.default('input')
    def _input(self):
        return ExpressionSelectionTextArea(df=self.df)

    @traitlets.default('adder')
    def _adder(self):
        return ColumnSelectionAdder(df=self.df, component=self.input)

    template = traitlets.Unicode('''
        <v-layout column>
            <component-input></component-input>
            <v-layout pa-4>
                <adder></adder>
            </v-layout>
        </v-layout>''').tag(sync=True)


class VirtualColumnEditor(v.VuetifyTemplate):
    df = traitlets.Any()
    editor = traitlets.Any()
    adder = traitlets.Any()
    on_close = traitlets.Any()
    column_name = traitlets.Unicode('mycolumn').tag(sync=True)
    components = traitlets.Dict(None, allow_none=True).tag(sync=True, **widgets.widget.widget_serialization)

    @traitlets.default('components')
    def _components(self):
        return {'editor': self.editor, 'adder': self.adder}
    @traitlets.default('editor')
    def _editor(self):
        return ExpressionTextArea(df=self.df, rows=1)
    @traitlets.default('adder')
    def _adder(self):
        return ColumnExpressionAdder(df=self.df, component=self.editor)

    template = traitlets.Unicode('''
        <v-layout column style="position: relative">
            <v-text-field placeholder="e.g. mycolumn" label="Column name" v-model='column_name' prepend-icon='edit'>test</v-text-field>
            <editor></editor>
            <div style="position: absolute; right: 20px; bottom: 30px; opacity: 0.8">
                <adder></adder>
            </div>
        </v-layout>''').tag(sync=True)

    def save_column(self):
        if self.editor.valid:
            self.df[self.column_name] = self.editor.v_model
            if self.on_close:
                self.on_close()


class ColumnList(v.VuetifyTemplate, vt.ColumnsMixin):
    column_filter = traitlets.Unicode('').tag(sync=True)
    valid_expression = traitlets.Bool(False).tag(sync=True)
    dialog_open = traitlets.Bool(False).tag(sync=True)
    editor = traitlets.Any()
    editor_open = traitlets.Bool(False).tag(sync=True)
    tooltip = traitlets.Unicode('Add example expression based on column...').tag(sync=True)
    template = traitlets.Unicode(load_template('columnlist.vue')).tag(sync=True)

    def __init__(self, df=None, **kwargs):
        super(ColumnList, self).__init__(df=df, **kwargs)
        traitlets.dlink((self.editor.editor, 'valid'), (self, 'valid_expression'))
        self.editor.editor.on_event('keypress.enter', self._on_enter)

    @traitlets.default('editor')
    def _editor(self):
        editor = VirtualColumnEditor(df=self.df)
        return editor

    @traitlets.default('components')
    def _components(self):
        return {'content-editor': self.editor}

    def _on_enter(self, *ignore):
        if self.valid_expression:
            self.editor.save_column()
            self.dialog_open = False

    def vue_add_virtual_column(self, data):
        self.dialog_open = True

    def vue_save_column(self, data):
        self.editor.save_column()
        self.dialog_open = False

    def vue_column_click(self, data):
        name = data['name']
        if name in self.df.virtual_columns:
            self.editor.editor.v_model = self.df.virtual_columns[name]
            self.editor.column_name = name
            self.dialog_open = True
