class Scheduler:

    def run_all(self, runtime, context):
        # future: async / celery / batching
        return runtime.execute(context)