define("vaex.ext.bqplot", ["@jupyter-widgets/base", "bqplot", "underscore"],
       function(widgets, bqplot, _) {
    "use strict";
/*
widgets.BoxView.prototype.remove_original = widgets.BoxView.prototype.remove
widgets.BoxView.prototype.remove = function() {
    console.log("remove monkey patch")
    if(this.children_views)
        this.children_views.remove()
    if(!this._is_removed) // protect against inf loop, if we monkey patch twice (during dev that can happen)
        this.remove_original()
    else
        this._is_removed = true
}*/

    var Image = bqplot.Mark.extend({

        render: function() {
            var base_render_promise = Image.__super__.render.apply(this);
            var el = this.d3el || this.el;
            window.last_el = el;
            window.last_image = this;
            this.im = el.append("image")
                .attr("class", "dot_img")
                .attr("xlink:href", this.model.get('src'))
                .attr("x", 0) //this.model.get('x'))
                .attr("y", 0) //this.model.get('y'))
                .attr("width", 1)//this.model.get('width'))
                .attr("height", 1)// this.model.get('height'))
                .attr("preserveAspectRatio", this.model.get('preserve_aspect_ratio'));

            this.width = this.parent.plotarea_width;
            this.height = this.parent.plotarea_height;
            this.map_id = widgets.uuid();
            this.display_el_classes = ["event_layer"];
            var that = this;
            this.event_metadata = {
            "mouse_over": {
                "msg_name": "hover",
                "lookup_data": false,
                "hit_test": true
            },
            "legend_clicked":  {
                "msg_name": "legend_click",
                "hit_test": true
            },
            "element_clicked": {
                "msg_name": "element_click",
                "lookup_data": false,
                "hit_test": true
            },
            "parent_clicked": {
                "msg_name": "background_click",
                "hit_test": false
            }}
            console.log('listen to...')
            this.is_removed = false;
            this.model.set("view_count", this.model.get("view_count")+1)
            this.model.save_changes()
            return base_render_promise.then(function() {
                console.log("promise...")
                that.event_listeners = {};
                that.create_listeners();
                console.log("drawing...")
                that.draw();
            });
        },
        remove: function() {
            var result = Image.__super__.remove.apply(this, arguments);
            console.log("remove")
            if(!this.is_removed) {
                console.log("removed")
                this.model.set("view_count", this.model.get("view_count")-1)
                this.model.save_changes()
                this.is_removed = true;
            }
            return result;
        },
        set_positional_scales: function() {
            var x_scale = this.scales.x,
                y_scale = this.scales.y;
            this.listenTo(x_scale, "domain_changed", function() {
                if (!this.model.dirty) {
                    var animate = true;
                    //this.update_xy_position(animate); }
                    this.set_ranges()
                 }
            });
            this.listenTo(y_scale, "domain_changed", function() {
                if (!this.model.dirty) {
                    var animate = true;
                    //this.update_xy_position(animate);
                    this.set_ranges()
                }
            });
        },
        set_ranges: function() {
                console.log("set ranges")
	            var x_scale = this.scales.x,
	                y_scale = this.scales.y
	                ;
	            if(x_scale) {
	                x_scale.set_range(this.parent.padded_range("x", x_scale.model));
	            }
	            if(y_scale) {
	                y_scale.set_range(this.parent.padded_range("y", y_scale.model));
	            }
	            var x_scale = this.scales.x, y_scale = this.scales.y;
	            var that = this;
	            var animation_duration = this.parent.model.get("animation_duration");// : 0;
                var el = this.d3el || this.el;
	            el.selectAll(".dot_img").transition()
	                .duration(animation_duration)
	                .attr("transform", function(d) {
                        var sx = x_scale.scale(1) - x_scale.scale(0);
                        var sy = y_scale.scale(1) - y_scale.scale(0);
                        var tx = x_scale.scale(that.model.get('x')) + x_scale.offset
                        var ty = y_scale.scale(that.model.get('y')) + y_scale.offset
                        sx *= that.model.get('width')
                        sy *= that.model.get('height')
                        //console.log(that.model)
                        //console.log(that.model.get('width'))
                        //console.log(sx)
                        //console.log(sy)
	                    return "translate(" + tx +
	                                    "," + ty + ") scale(" + sx + ", " + sy + ")"});
	            //this.x_pixels = this.model.mark_data.map(function(el) { return x_scale.scale(el.x) + x_scale.offset; });
	            //this.y_pixels = this.model.mark_data.map(function(el) { return y_scale.scale(el.y) + y_scale.offset; });
        },

        create_listeners: function() {
            Image.__super__.create_listeners.apply(this);

            this.listenTo(this.model, "change:src", this.update_src, this);
            this.listenTo(this.model, "change:x", this.update_x, this);
            this.listenTo(this.model, "change:y", this.update_y, this);
            this.listenTo(this.model, "change:width", this.update_width, this);
            this.listenTo(this.model, "change:height", this.update_height, this);
            this.listenTo(this.model, "change:preserve_aspect_ratio", this.update_preserve_aspect_ratio, this);
            this.listenTo(this.parent, "bg_clicked", function() {
                        this.event_dispatcher("parent_clicked");
                    });
        },

        update_xy_position: function(animate) {
            //console.log("update_xy_position");
        },

        update_src: function(model, new_x) {
          //this.im.attr('x', new_x);
          //console.log("update src")
            this.im.attr("xlink:href", this.model.get('src'))
            //this.set_ranges();
        },
        update_x: function(model, new_x) {
          //this.im.attr('x', new_x);
            this.set_ranges();
        },

        update_y: function(model, new_y) {
          //this.im.attr('y', new_y);
            this.set_ranges();
        },

        update_width: function(model, new_width) {
          //this.im.attr('width', new_width);
            this.set_ranges();
        },

        update_height: function(model, new_height) {
          //this.im.attr('height', new_height);
            this.set_ranges();
        },

        update_preserve_aspect_ratio: function(model, new_preserve_aspect_ratio) {
          this.im.attr('preserveAspectRatio', new_preserve_aspect_ratio);
        },

        draw: function() {
            console.log("in draw")
            this.set_ranges();
            this.im.on("click", _.bind(function(d, i) {
            this.event_dispatcher("element_clicked",
			      {"data": d, "index": 0});
            }, this));

        },

    });


   var ImageModel = bqplot.MarkModel.extend({
        defaults: function() {
            var more_defaults =  {
                    _model_name: "ImageModel",
                    _view_name: "Image",
                    _view_module: "vaex.ext.bqplot",
                    _model_module: "vaex.ext.bqplot",
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    view_count: 0,
                }
            return _.extend(bqplot.MarkModel.prototype.defaults(), more_defaults);
        },
        initialize: function() {
            ImageModel.__super__.initialize.apply(this, arguments);
        },

    });



    console.log("blaat")
	    return {
	        ImageModel: ImageModel,
	        Image:Image,
	    };
})
//require(["vaex.ext.bqplot"])