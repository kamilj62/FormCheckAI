def run_plugins(plugins, context):
    outputs = []

    for name, plugin in plugins.items():
        try:
            outputs.append(plugin.run(context))
        except Exception as e:
            outputs.append({
                "label": "error",
                "confidence": 0.0,
                "source": name,
                "error": str(e)
            })

    return outputs