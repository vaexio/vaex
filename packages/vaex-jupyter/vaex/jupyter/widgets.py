from __future__ import absolute_import

import os

import ipyvuetify as v
import ipywidgets as widgets
import traitlets
from traitlets import *  # noqa
from traitlets import Dict, observe

import vaex.jupyter

from . import traitlets as vt


def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


vaex_components = {}


def component(name):
    def wrapper(cls):
        vaex_components[name] = cls
        return cls
    return wrapper


# mixin class
class UsesVaexComponents(traitlets.HasTraits):
    @traitlets.default('components')
    def _components(self):
        return vaex_components


class PlotTemplate(v.VuetifyTemplate):
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


@component('vaex-counter')
class Counter(v.VuetifyTemplate):
    characters = traitlets.List(traitlets.Unicode()).tag(sync=True)
    value = traitlets.Integer(None, allow_none=True)
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


@component('vaex-status')
class Status(v.VuetifyTemplate):
    value = traitlets.Unicode().tag(sync=True)
    template = traitlets.Unicode('''
          <v-slide-y-transition leave-absolute>
              <span :key="value" v-html='value'></span>
          </v-slide-y-transition>
      ''').tag(sync=True)


@component('vaex-progress-circular')
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


@component('vaex-expression')
class Expression(v.TextField):
    df = traitlets.Any()
    valid = traitlets.Bool(True)
    value = vt.Expression(None, allow_none=True)

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

    @traitlets.default('value')
    def _value(self):
        self.value = None if self.v_model is None else self.df[self.v_model]

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
    def _on_update_v_model(self, change):
        self.check_expression()

    @traitlets.observe('value')
    def _on_update_value(self, change):
        self.v_model = None if self.value is None else str(self.value)

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
        self.value = self.v_model
        self._clear_succes()
        return True

    @vaex.jupyter.debounced(delay_seconds=1.5, skip_gather=True)
    def _clear_succes(self):
        self.success_messages = None


ExpressionTextArea = Expression


class ExpressionSelectionTextArea(ExpressionTextArea):
    # selection is v_model
    selection_name = traitlets.Any('default')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.update_selection()

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


@component('vaex-selection-editor')
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


class Selection(v.VuetifyTemplate):
    df = traitlets.Any().tag(sync_ref=True)
    name = traitlets.Unicode('default').tag(sync=True)
    value = traitlets.Unicode(None, allow_none=True).tag(sync=True)

    @traitlets.default('template')
    def _template(self):
        return load_template('vue/selection.vue')

    @traitlets.default('components')
    def _components(self):
        return vaex_components


class SelectionToggleList(v.VuetifyTemplate):
    df = traitlets.Any().tag(sync_ref=True)
    title = traitlets.Unicode('Choose selections').tag(sync=True)
    selection_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    value = traitlets.List(traitlets.Unicode()).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df.signal_selection_changed.connect(self._on_change_selection)

    def _on_change_selection(self, df, name):
        new_names = [name for name in self.df.selection_histories.keys() if not name.startswith('__') and df.has_selection(name)]
        self.selection_names = new_names
        self.value = [v for v in self.value if v in self.selection_names]

    @traitlets.default('selection_names')
    def _selection_names(self):
        return [name for name in self.df.selection_histories.keys() if not name.startswith('__')]

    @traitlets.default('template')
    def _template(self):
        return load_template('vue/selection_toggle_list.vue')

    @traitlets.default('components')
    def _components(self):
        return vaex_components




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
    _metadata = traitlets.Dict(default_value=None, allow_none=True).tag(sync=True)
    column_filter = traitlets.Unicode('').tag(sync=True)
    valid_expression = traitlets.Bool(False).tag(sync=True)
    dialog_open = traitlets.Bool(False).tag(sync=True)
    editor = traitlets.Any()
    editor_open = traitlets.Bool(False).tag(sync=True)
    tooltip = traitlets.Unicode('Add example expression based on column...').tag(sync=True)
    template = traitlets.Unicode(load_template('vue/columnlist.vue')).tag(sync=True)

    def __init__(self, df, **kwargs):
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


class ColumnPicker(v.VuetifyTemplate, vt.ColumnsMixin):
    template = traitlets.Unicode(load_template('vue/column-select.vue')).tag(sync=True)
    label = traitlets.Unicode('Column').tag(sync=True)
    value = vt.Expression(None, allow_none=True).tag(sync=True)


tools_items_default = [
    {'value': 'pan-zoom', 'icon': 'pan_tool', 'tooltip': "Pan & zoom"},
    {'value': 'select-rect', 'icon': 'mdi-selection-drag', 'tooltip': "Rectangle selection"},
    {'value': 'select-x', 'icon': 'mdi-drag-vertical', 'tooltip': "X-Range selection"},
]

selection_items_default = [
    {'value': 'replace', 'icon': 'mdi-circle-medium', 'tooltip': "Replace mode"},
    {'value': 'and', 'icon': 'mdi-set-center', 'tooltip': "And mode"},
    {'value': 'or', 'icon': 'mdi-set-all', 'tooltip': "Or mode"},
    {'value': 'subtract', 'icon': 'mdi-set-left', 'tooltip': "Subtract mode"},
]

transform_items_default = ['identity', 'log', 'log10', 'log1p', 'log1p']


class ToolsSpeedDial(v.VuetifyTemplate):
    expand = traitlets.Bool(False).tag(sync=True)
    value = traitlets.Unicode(tools_items_default[0]['value'], allow_none=True).tag(sync=True)
    items = traitlets.Any(tools_items_default).tag(sync=True)
    template = traitlets.Unicode(load_template('vue/tools-speed-dial.vue')).tag(sync=True)
    children = traitlets.List().tag(sync=True, **widgets.widget_serialization)

    def vue_action(self, data):
        self.value = data['value']


class ToolsToolbar(v.VuetifyTemplate):
    interact_value = traitlets.Unicode(tools_items_default[0]['value'], allow_none=True).tag(sync=True)
    interact_items = traitlets.Any(tools_items_default).tag(sync=True)
    transform_value = traitlets.Unicode(transform_items_default[0]).tag(sync=True)
    transform_items = traitlets.List(traitlets.Unicode(), default_value=transform_items_default).tag(sync=True)
    supports_transforms = traitlets.Bool(True).tag(sync=True)

    supports_normalize = traitlets.Bool(True).tag(sync=True)
    z_normalize = traitlets.Bool(False, allow_none=True).tag(sync=True)
    normalize = traitlets.Bool(False).tag(sync=True)

    selection_mode_items = traitlets.Any(selection_items_default).tag(sync=True)
    selection_mode = traitlets.Unicode('replace').tag(sync=True)

    @traitlets.default('template')
    def _template(self):
        return load_template('vue/tools-toolbar.vue')

    @observe('z_normalize')
    def _observe_normalize(self, change):
        self.normalize = bool(self.z_normalize)

class VuetifyTemplate(v.VuetifyTemplate):
    _metadata = traitlets.Dict(default_value=None, allow_none=True).tag(sync=True)

class ContainerCard(v.VuetifyTemplate):
    _metadata = Dict(default_value=None, allow_none=True).tag(sync=True)
    @traitlets.default('template')
    def _template(self):
        return load_template('vue/card.vue')
    title = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    subtitle = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    text = traitlets.Unicode(None, allow_none=True).tag(sync=True)
    main = traitlets.Any().tag(sync=True, **widgets.widget_serialization)
    controls = traitlets.List().tag(sync=True, **widgets.widget_serialization)
    card_props = traitlets.Dict().tag(sync=True)
    main_props = traitlets.Dict().tag(sync=True)
    show_controls = traitlets.Bool(False).tag(sync=True)

class Html(v.Html):
    _metadata = traitlets.Dict(default_value=None, allow_none=True).tag(sync=True)


class LinkList(VuetifyTemplate):
    items = traitlets.List(
        [
            {'title': 'Vaex (data aggregation)', 'url': "https://github.com/vaexio/vaex", 'img': 'https://vaex.io/img/logos/logo-grey.svg', },
            {'icon': "dashboard", 'title': "Voila (dashboard)", 'url': "https://github.com/voila-dashboards/voila"},
            {'icon': "mdi-database", 'title': "DataFrame server", 'url': "http://dataframe.vaex.io/"},
            {'title': 'ipyvolume (3d viz)', 'url': "https://github.com/maartenbreddels/ipyvolume", 'img': 'https://raw.githubusercontent.com/maartenbreddels/ipyvolume/master/misc/icon.svg', },
            {'title': 'GitHub Repo', 'url': 'https://github.com/vaexio/vaex', 'img': 'https://github.githubassets.com/pinned-octocat.svg'},
            {'icon': "widgets", 'title': "jupyter widgets", 'url': "https://github.com/jupyter-widgets/ipywidgets"},
        ],
    ).tag(sync=True)
    @traitlets.default('template')
    def _template(self):
        return load_template('vue/link-list.vue')


import ipyvuetify as v
import traitlets


class SettingsEditor(v.VuetifyTemplate):
    template_file = os.path.join(os.path.dirname(__file__), "vue/vjsf.vue")

    vjsf_loaded = traitlets.Bool(False).tag(sync=True)
    values = traitlets.Dict(default_value={}).tag(sync=True)
    schema = traitlets.Dict().tag(sync=True)
    valid = traitlets.Bool(False).tag(sync=True)


def watch():
    import ipyvue
    ipyvue.watch(os.path.dirname(__file__) + "/vue")
