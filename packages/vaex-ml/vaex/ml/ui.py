import traitlets
from . import generate
import ipywidgets as widgets
from vaex.serialize import register


@register
class Widgetize(widgets.VBox):
    def __init__(self, pipeline_object, dataframe):
        '''
        Docstring
        :params:
        :param pipeline_object: the vaex pipeline element object
        :param dataframe: a vaex DataFrame on which the pipeline object will operate
        '''
        super(Widgetize, self).__init__()

        self.pipeline_object = pipeline_object
        self.dataframe = dataframe

        ui_dict_ = {
            'Checkbox': widgets.Checkbox,
            'Text': widgets.Text,
            'IntText': widgets.IntText,
            'FloatText': widgets.FloatText,
            'Button': widgets.Button,
            'SelectMultiple': widgets.SelectMultiple,
            'FloatRangeSlider': widgets.FloatRangeSlider,
            'HTML': widgets.HTML,
        }

        widget_list = []
        for trait_name in pipeline_object.trait_names():
            metadata = pipeline_object.traits()[trait_name].metadata
            # only traits with ui metadata will show up in the ui
            if 'ui' in metadata:
                widget_class = ui_dict_[metadata['ui']]
                widget = widget_class(description=trait_name)
                widget_list.append(widget)

                if isinstance(widget, widgets.SelectMultiple):
                    widget.options = self.dataframe.get_column_names(virtual=True, strings=True)
                    traitlets.link((self.pipeline_object, trait_name), (widget, 'value'))
                else:
                    traitlets.link((self.pipeline_object, trait_name), (widget, 'value'))

        fit_button = widgets.Button(description='Fit', button_style='info')
        fit_button.on_click(lambda button: self.pipeline_object.fit(self.dataframe))
        # add it to the 'form'.
        widget_list.append(fit_button)

        # Adding all the  widget elements to the parent object
        self.children = widget_list
