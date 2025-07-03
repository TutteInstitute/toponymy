# Widgets Folder

This folder contains jupyter notebook widgets. Each widget can be built separately by navigating into its subdirectory and running the standard build commands.
These widgets are pre-built, meaning they should work out of the box for most users. We include the source files and build instructions for other contributers who may want to extend/improve the existing widgets.

---

## Building a Widget

To build a specific widget, follow these steps:

1. Navigate to the widget's subdirectory:

```bash
    cd widgets/<widget_name>
```

2. Install dependencies:

```bash
    npm install
``` 

3. Build the widget:

```bash
    npm run build
``` 

This will produce the bundled widget files in the `widgets/<widget_name>/dist/` folder, ready for use by anywidget.

## Including the Widget in Python

To include the widget you need to include the js/css code in an `anywidget.AnyWidget` class. This will look something like:

```python
class MyWidget(anywidget.AnyWidget):
    # point widget to your dist folder
    _esm = pathlib.Path(__file__).parent / "widgets/myWidget/dist/widget.js"
    _css = pathlib.Path(__file__).parent / "widgets/myWidget/dist/widget.css"

    # send python variables to js with traitlets
    data = traitlets.Dict(default_value={}).tag(sync=True) 
```

You can find the existing widget classes in the `toponymy/plotting.py` folder.

## Notes

- Make sure you have Node.js and npm installed before running these commands.

