class PluginRuntime:

    def __init__(self, plugins):
        self.plugins = plugins

    def execute(self, context):
        results = {}

        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.run(context)
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "source": name
                }

        return results