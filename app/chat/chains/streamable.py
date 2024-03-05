from queue import Queue
from threading import Thread
from app.chat.callbacks.stream import StreamingHandler
from flask import current_app


# Instead of subclassing like we do in test.py, we are creating
# a mixin. This allows us to easily subclass different chains
# with streaming capability
class StreamableChain:
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task(app_context):
            # flask context
            app_context.push()
            # Assign our callbacks when calling our chain
            # with the user scoped handler
            self(input, callbacks=[handler])

        # We need to pass the flask app_context to the thread to avoid
        # Error:
        #       raise RuntimeError(unbound_message) from None
        #       RuntimeError: Working outside of application context.
        Thread(target=task, args=[current_app.app_context()]).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token
