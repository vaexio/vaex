# Vaex server
## Why

There are various cases where the calculations and/or aggregations need to happen on a different computer than where the (aggregated) data is needed. For instance, when making a dashboard, the dashboard server might not be powerful enough for the calculations. Another example is where the client lives in a different process, such as a browser.

## Starting the dataframe server


```{admonition} Use our server first
:class: tip

You can skip running your own server and first try out using [https://dataframe.vaex.io](https://dataframe.vaex.io)
```

The vaex (web) server can be started from the command line like:
```bash
$ vaex server --port 8082 /data/taxi/yellow_taxi_2012.hdf5 gaia=/data/gaia/gaia-edr3-x-ps1.hdf5
INFO:MainThread:vaex.server:yellow_taxi_2012:  http://0.0.0.0:8082/dataset/yellow_taxi_2012 for REST or ws://0.0.0.0:8082/yellow_taxi_2012 for websocket
INFO:MainThread:vaex.server:gaia:  http://0.0.0.0:8082/dataset/gaia for REST or ws://0.0.0.0:8082/gaia for websocket
INFO:     Started server process [617048]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8082 (Press CTRL+C to quit)
```

Pass files on the command line, or query help by passing the `--help` flag.

## Python API

When the client is a Python program, the easiest API is the remote dataframe in the `vaex` packages itself. This does not use the REST API, but communicates over a websocket for low latency bi-directional communication.

```python
import vaex
# the data is kept remote
df = vaex.open('vaex+wss://dataframe.vaex.io/example')
# only the result of the aggregations are send over the wire
df.x.mean()
```

This means you can use almost all features of a normal (local) Vaex dataframe, without having to download the data.


## REST API

When the client is non-Python, or when you want to avoid the `vaex` dependency, the REST API can be used.

A Vaex server is running at `dataframe.vaex.io` and it's API documentation can be browsed at [https://dataframe.vaex.io/docs](https://dataframe.vaex.io/docs)

Some endpoints can be easily queries using curl
```bash
$ curl -i https://dataframe.vaex.io/histogram/example/x\?shape\=16
HTTP/1.1 200 OK
Server: nginx/1.18.0 (Ubuntu)
Date: Thu, 01 Apr 2021 11:23:16 GMT
Content-Type: application/json
Content-Length: 430
Connection: keep-alive
x-process-time: 0.03632664680480957
x-data-passes: 2

{"dataset_id":"example","centers":[-71.61332178115845,-58.57391309738159,-45.534504413604736,-32.49509572982788,-19.455687046051025,-6.41627836227417,6.6231303215026855,19.66253900527954,32.7019476890564,45.74135637283325,58.78076505661011,71.82017374038696,84.85958242416382,97.89899110794067,110.93839979171753,123.97780847549438],"values":[3.0,0.0,3.0,917.0,13706.0,154273.0,147171.0,12963.0,960.0,1.0,1.0,0.0,0.0,0.0,1.0,1.0]}
```

While the `POST` method might be more convenient from a Javascript client or using a HTTP Library.

### Python using requests

[Requests](https://docs.python-requests.org/en/master/) is an easy to use HTTP library.
```python
import requests
data = {
    'dataset_id': 'gaia-dr2',
    'expression_x': 'l',
    'expression_y': 'b',
    'filter': None,
    'virtual_columns': [],
    'min_x': 0,
    'max_x': 360,
    'min_y': -90,
    'max_y': 90,
    'shape_x': 512,
    'shape_y': 256,
}
response = requests.post('https://dataframe.vaex.io/heatmap', json=data)
response.json()
assert response.status_code == 200, 'oops, something went wrong'
```

```python
{'dataset_id': 'gaia-dr2',
 'centers_x': [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5],
 'centers_y': [-67.5, -22.5, 22.5, 67.5],
 'values': [[3508786.0,
   2711710.0,
   2287021.0,
   2042114.0,
   2009057.0,
   2448207.0,
   3716951.0,
   3644323.0],
  [250883466.0,
   100064757.0,
   49538929.0,
   28273970.0,
   30521201.0,
   53214391.0,
   159460735.0,
   251170124.0],
  [166984543.0,
   110774989.0,
   43475771.0,
   31343345.0,
   31584354.0,
   44061582.0,
   108436851.0,
   189699927.0],
  [3388522.0,
   2848641.0,
   2221241.0,
   1997993.0,
   1941090.0,
   2215271.0,
   3061986.0,
   3387287.0]]}
```

```javascript
var inputData = {
    dataset_id: 'gaia-dr2',
    expression_x: 'l',
    expression_y: 'b',
    filter: null,
    virtual_columns: [],
    min_x: 0,
    max_x: 360,
    min_y: -90,
    max_y: 90,
    shape: [512, 256],
};
var result = await fetch("https://dataframe.vaex.io/heatmap", {method: 'POST', body: JSON.stringify(inputData)})
var data = await result.json();
console.log(data);
```

```
{dataset_id: "gaia-dr2", centers_x: Array(512), centers_y: Array(256), values: Array(256)}
centers_x: (512) [0.3515625, 1.0546875, 1.7578125, 2.4609375, 3.1640625, 3.8671875, 4.5703125, 5.2734375, 5.9765625, 6.6796875, 7.3828125, 8.0859375, 8.7890625, 9.4921875, 10.1953125, 10.8984375, 11.6015625, 12.3046875, 13.0078125, 13.7109375, 14.4140625, 15.1171875, 15.8203125, 16.5234375, 17.2265625, 17.9296875, 18.6328125, 19.3359375, 20.0390625, 20.7421875, 21.4453125, 22.1484375, 22.8515625, 23.5546875, 24.2578125, 24.9609375, 25.6640625, 26.3671875, 27.0703125, 27.7734375, 28.4765625, 29.1796875, 29.8828125, 30.5859375, 31.2890625, 31.9921875, 32.6953125, 33.3984375, 34.1015625, 34.8046875, 35.5078125, 36.2109375, 36.9140625, 37.6171875, 38.3203125, 39.0234375, 39.7265625, 40.4296875, 41.1328125, 41.8359375, 42.5390625, 43.2421875, 43.9453125, 44.6484375, 45.3515625, 46.0546875, 46.7578125, 47.4609375, 48.1640625, 48.8671875, 49.5703125, 50.2734375, 50.9765625, 51.6796875, 52.3828125, 53.0859375, 53.7890625, 54.4921875, 55.1953125, 55.8984375, 56.6015625, 57.3046875, 58.0078125, 58.7109375, 59.4140625, 60.1171875, 60.8203125, 61.5234375, 62.2265625, 62.9296875, 63.6328125, 64.3359375, 65.0390625, 65.7421875, 66.4453125, 67.1484375, 67.8515625, 68.5546875, 69.2578125, 69.9609375, …]
centers_y: (256) [-89.6484375, -88.9453125, -88.2421875, -87.5390625, -86.8359375, -86.1328125, -85.4296875, -84.7265625, -84.0234375, -83.3203125, -82.6171875, -81.9140625, -81.2109375, -80.5078125, -79.8046875, -79.1015625, -78.3984375, -77.6953125, -76.9921875, -76.2890625, -75.5859375, -74.8828125, -74.1796875, -73.4765625, -72.7734375, -72.0703125, -71.3671875, -70.6640625, -69.9609375, -69.2578125, -68.5546875, -67.8515625, -67.1484375, -66.4453125, -65.7421875, -65.0390625, -64.3359375, -63.6328125, -62.9296875, -62.2265625, -61.5234375, -60.8203125, -60.1171875, -59.4140625, -58.7109375, -58.0078125, -57.3046875, -56.6015625, -55.8984375, -55.1953125, -54.4921875, -53.7890625, -53.0859375, -52.3828125, -51.6796875, -50.9765625, -50.2734375, -49.5703125, -48.8671875, -48.1640625, -47.4609375, -46.7578125, -46.0546875, -45.3515625, -44.6484375, -43.9453125, -43.2421875, -42.5390625, -41.8359375, -41.1328125, -40.4296875, -39.7265625, -39.0234375, -38.3203125, -37.6171875, -36.9140625, -36.2109375, -35.5078125, -34.8046875, -34.1015625, -33.3984375, -32.6953125, -31.9921875, -31.2890625, -30.5859375, -29.8828125, -29.1796875, -28.4765625, -27.7734375, -27.0703125, -26.3671875, -25.6640625, -24.9609375, -24.2578125, -23.5546875, -22.8515625, -22.1484375, -21.4453125, -20.7421875, -20.0390625, …]
dataset_id: "gaia-dr2"
values: (256) [ …]
__proto__: Object
```

<script src="https://unpkg.com/underscore@1.8.3"></script>

## Example using plotly.js


Combining the previous with the [plotly.js library](https://plotly.com/javascript/getting-started/) we can make an interactive plot:

### Sky map

First, make sure we have a div
```html
<div id="plotlyHeatmap"></div>
```

<div id="plotlyHeatmap"></div>


Then load the data, and plot it using plotly.js:
```javascript


var skyMapInput = {
    dataset_id: 'gaia-dr2',
    expression_x: 'l',
    expression_y: 'b',
    virtual_columns: {
        distance: "1/parallax"
    },
    filter: this.filter,
    min_x: 0,
    max_x: 360,
    min_y: -90,
    max_y: 90,
    shape: [512, 256],
};

async function loadData(heatmapInput) {
    const result = await fetch("https://dataframe-dev.vaex.io/heatmap", {method: 'POST', body: JSON.stringify(heatmapInput)})
    const data = await result.json();
    return data;
}

function plotData(elementId, data, log, xaxis, yaxis) {
    const trace_data = {
        x: data.centers_x,
        y: data.centers_y,
        z: log ? data.values.map((ar1d) => ar1d.map(Math.log1p)) : data.values,
        type: 'heatmap',
        colorscale: 'plasma',
        transpose: true,
    };
    var layout = {
        xaxis: {
            title: {
                text: data.expression_x,
            },
            ...xaxis
        },
        yaxis: {
            title: {
                text: data.expression_y,
            },
            ...yaxis
        }
    };
    Plotly.react(elementId, [trace_data], layout);
}

async function plot(elementId, heatmapInput, xaxis, yaxis) {
    const heatmapOutput = await loadData(heatmapInput);
    await plotData(elementId, heatmapOutput, true, xaxis, yaxis);
}

plot('plotlyHeatmap', skyMapInput);
```

Adding an event handler, will refine the data when we zoom in:
```javascript

function addZoomHandler(elementId, heatmapInput) {
    document.getElementById(elementId).on('plotly_relayout', async (e) => {
        // mutate input data
        heatmapInput.min_x = e["xaxis.range[0]"]
        heatmapInput.max_x = e["xaxis.range[1]"]
        heatmapInput.min_y = e["yaxis.range[0]"]
        heatmapInput.max_y = e["yaxis.range[1]"]
        // and plot again
        plot(elementId, heatmapInput);
    })
}


```

```{include} ../data/rest/sky.html
```


<script>

var skyMapInput = {
    dataset_id: 'gaia-dr2',
    expression_x: 'l',
    expression_y: 'b',
    virtual_columns: {
        distance: "1/parallax"
    },
    filter: this.filter,
    min_x: 0,
    max_x: 360,
    min_y: -90,
    max_y: 90,
    shape: [512, 256],
};

async function loadData(heatmapInput) {
    const result = await fetch("https://dataframe-dev.vaex.io/heatmap", {method: 'POST', body: JSON.stringify(heatmapInput)})
    const data = await result.json();
    return data;
}

function plotData(elementId, data, log, xaxis, yaxis) {
    const trace_data = {
        x: data.centers_x,
        y: data.centers_y,
        z: log ? data.values.map((ar1d) => ar1d.map(Math.log1p)) : data.values,
        type: 'heatmap',
        colorscale: 'plasma',
        transpose: true,
    };
    var layout = {
        xaxis: {
            title: {
                text: data.expression_x,
            },
            ...xaxis
        },
        yaxis: {
            title: {
                text: data.expression_y,
            },
            ...yaxis
        }
    };
    Plotly.react(elementId, [trace_data], layout);
}

async function plot(elementId, heatmapInput, xaxis, yaxis) {
    const heatmapOutput = await loadData(heatmapInput);
    await plotData(elementId, heatmapOutput, true, xaxis, yaxis);
}

function addZoomHandler(elementId, heatmapInput) {
    document.getElementById(elementId).on('plotly_relayout', async (e) => {
        // mutate input data
        heatmapInput.min_x = e["xaxis.range[0]"]
        heatmapInput.max_x = e["xaxis.range[1]"]
        heatmapInput.min_y = e["yaxis.range[0]"]
        heatmapInput.max_y = e["yaxis.range[1]"]
        // and plot again
        plot(elementId, heatmapInput);
    })
}

requirejs(['https://cdn.plot.ly/plotly-1.58.4.min.js'], (Plotly) => {
    window.Plotly = Plotly;
    (async () => {
        // to update the sky.html data, uncomment the next line, comment the line after
        // and copy paste the reply in sky.html
        // await plot('plotlyHeatmap', skyMapInput);
        await plotData('plotlyHeatmap', skyMapOutput, true);
        addZoomHandler('plotlyHeatmap', skyMapInput);
    })();
});

</script>

### CMD

We can now easily add a second heatmap
```html
<div id="plotlyHeatmapCMD"></div>
```

And plot a different heatmap (a color-magnitude diagram) on this div.
```javascript

var cmdInput = {
    dataset_id: 'gaia-dr2',
    expression_x: 'phot_bp_mean_mag-phot_rp_mean_mag',
    expression_y: 'M_g',
    virtual_columns: {
        distance: "1/parallax",
        M_g: "phot_g_mean_mag-(5*log10(distance)+10)"
    },
    filter: '((pmra**2+pmdec**2)<100)&(parallax_over_error>10)&(abs(b)>20)',
    min_x: -1,
    max_x: 5,
    min_y: 15,
    max_y: -5,
    shape_x: 256,
    shape_y: 256,
};

async () => {
    await plot('plotlyHeatmapCMD', cmdInput);
    addZoomHandler('plotlyHeatmapCMD', cmdInput);
}

```

```{include} ../data/rest/cmd.html
```


<div id="plotlyHeatmapCMD"></div>


<script>

var cmdInput = {
    dataset_id: 'gaia-dr2',
    expression_x: 'phot_bp_mean_mag-phot_rp_mean_mag',
    expression_y: 'M_g',
    virtual_columns: {
        distance: "1/parallax",
        M_g: "phot_g_mean_mag-(5*log10(distance)+10)"
    },
    filter: '((pmra**2+pmdec**2)<100)&(parallax_over_error>10)&(abs(b)>20)',
    min_x: -1,
    max_x: 5,
    min_y: 15,
    max_y: -5,
    shape_x: 256,
    shape_y: 256,
};

requirejs(['https://cdn.plot.ly/plotly-1.58.4.min.js'], (Plotly) => {
    window.Plotly = Plotly;
    (async () => {
        // await plot('plotlyHeatmapCMD', cmdInput);
        plotData('plotlyHeatmapCMD', heatmapOutput, true, {}, {autorange: 'reversed'});
        addZoomHandler('plotlyHeatmapCMD', cmdInput);
    })();
});

</script>