require.undef('hello');

define('hello', ["jupyter-js-widgets", "underscore"], function(widgets, _) {
    console.log("define view")
    // Define the HelloView
    var JsonSourceView = widgets.WidgetView.extend({
        render: function() {
            console.log("init: " +this)
            console.log("model url: " +this.model.get("url"))
        }
    });
    console.log("define model")
    // Define the HelloView
    var JsonSourceModel = widgets.WidgetModel.extend({
        defaults: _.extend({}, widgets.WidgetModel.prototype.defaults, {
            _model_name: "JsonSourceModel",
            _view_name: "JsonSourceView",
            _view_module: "hello",
            _model_module: "hello",
            url: "ajax",
            method: "POST",
            data_input: {},
            data_input_static: {},
            listeners: [],
            custom_json:"return data",
            custom_js_output: "",
            scope: {}
        }),
        initialize: function() {
            JsonSourceModel.__super__.initialize.apply(this);
            console.log(this)
            console.log("init: " +this)
            console.log("url: " +this.get("url"))
            console.log(this.get('listeners'))
            this.previous_listeners = null;
            var obj = this.on('change:custom_json', function(model, value) {
                //console.log("changed json!!!!!!" + value)
                //console.log(">>>" +this.get('custom_json'))
            }, this);
            var obj = this.on('change:listeners', function(model, value) {
                //console.log("changed!!!" + arguments)
                //console.log(this.previous("listeners"))
                //console.log(this.get("listeners"))
                this.previous_listeners = value;
            }, this);
            this._bind_listeners()
            console.log(obj)
            var that = this;
            this.trigger_ajax = _.debounce( function() {
                console.log("do debounched ajax call");
                that._real_trigger_ajax()
            }, 250);
        },
        _bind_listeners: function() {
            var listeners = this.get("listeners");
            _.each(listeners, function(listener) {
                    var widget = listener[0];
                    var attribute = listener[1];
                    var key_store = listener[2];
                    console.log("listen to" +widget.name +"." + attribute +" and store as " +key_store)
                    this.get('data_input')[key_store] = widget.get(attribute);
                    widget.on("change:" + attribute, function(model, value) {
                        console.log(key_store)
                        this.get('data_input')[key_store] = value;
                        //console.log("store " + value +" as " +key_store)
                        this.dirty = true;
                        this.trigger_ajax()
                    }, this);
                }, this);
        },
        _real_trigger_ajax: function() {
            console.log("debounched ajax call");
            var payload = this.get('data_input');
            console.log(this)
            var fn = Function("data", "data_static", this.get('custom_json'))
            console.log(this.get('data_input_static'))
            payload = fn(payload, this.get('data_input_static'))
            //payload = $.map(payload, JSON.stringify)
            var payload_str = {}
            for(var key in payload) {
                payload_str[key] = JSON.stringify(payload[key])
            }
            console.log(payload_str)
            $.ajax(this.get('url'), {
                context: this,
                data: (payload_str),
                dataType: "json",
                error: function() {
                    console.log("ajax error...")
                },
                method: this.get("method"),
                success: function(data, status) {
                    this._succes(data, status)
                }
                
            })
        },
        _succes: function(data, status) {
            console.log(this.get('scope'))
            var fn = Function("data", "scope", "data_input", "data_input_static", this.get('custom_js_output'))
            fn(data, this.get('scope'), this.get('data_input'), this.get('data_input_static'))
        }
    }, {
        serializers: _.extend({
            listeners: {deserialize: widgets.unpack_models},
            source: {deserialize: widgets.unpack_models},
            scope:  {deserialize: widgets.unpack_models},
        }, widgets.WidgetModel.serializers),
    });

    return {
        JsonSourceView: JsonSourceView,
        JsonSourceModel: JsonSourceModel
    }
});
require(["hello"])
