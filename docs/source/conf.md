
# Configuration

All settings in Vaex can be configured in a uniform way, based on [Pydantic](https://pydantic-docs.helpmanual.io/usage/settings/). From a Python runtime, configuration of settings can be done via the `vaex.settings` module.
```python
import vaex
vaex.settings.main.thread_count = 10
vaex.settings.display.max_columns = 50
```

Via environmental variables:
```
$ VAEX_NUM_THREADS=10 VAEX_DISPLAY_MAX_COLUMNS=50 python myservice.py
```

Otherwise, values are obtained from a `.env` [file using dotenv](https://saurabh-kumar.com/python-dotenv/#usages) from the current workding directory.
```
VAEX_NUM_THREADS=22
VAEX_CHUNK_SIZE_MIN=2048
```

Lastly, a global yaml file from `$VAEX_PATH_HOME/.vaex/main.yaml` is loaded (with last priority).
```
thread_count: 33
display:
  max_columns: 44
  max_rows: 20
```

If we now run `vaex settings yaml`, we see the effective settings as yaml output:
```
$ VAEX_NUM_THREADS=10 VAEX_DISPLAY_MAX_COLUMNS=50 vaex settings yaml
...
chunk:
  size: null
  size_min: 2048
  size_max: 1048576
display:
  max_columns: 50
  max_rows: 20
thread_count: 10
...
```


## Developers

When updating `vaex/settings.py`, run the `vaex settings watch` to generate this documentation below automatically when saving the file.

## Schema

A JSON schema can be generated using

```
$ vaex settings schema > vaex-settings.schema.json
```

<!-- autogenerate markdown below -->

If you see this you should run `vaex settings watch` or `vaex settings docgen`.
