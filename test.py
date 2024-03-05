from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from queue import Queue
from threading import Thread


load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs):
        print(f"There was an error {error}")
        self.queue.put(None)


# Controls how OpenAI responds to Langchain
# This will make OpenAI to stream to Langchain but we
# wont see anything as Langchain does not stream becasue of this flag
chat = ChatOpenAI(streaming=True)

prompt = ChatPromptTemplate.from_messages([("human", "{content}")])

# LLMChain will not stream the response, even with the stream method
# You must override the stream method if you want streaming
# chain = LLMChain(llm=chat, prompt=prompt)


# messages = prompt.format_messages(content="tell me a joke")
#
# print(messages)
#
# # How we call our model controls how OpenAI responds to Langchain
# # AND controls how Langchain responds to us
# # chat.stream overrides ChatOpenAI(streaming=True)
# # outputs a generator
# for message in chat.stream(messages):
#     print(message.content)


# To get streaming to work we subclass it and override the stream
# method
class StreamingChain(LLMChain):
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue)

        def task():
            # Assign our callbacks when calling our chain
            # with the user scoped handler
            self(input, callbacks=[handler])

        # Start this in parallel on separate thread
        Thread(target=task).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token


chain = StreamingChain(llm=chat, prompt=prompt)

for output in chain.stream(input={"content": "tell me a joke"}):
    print(output)
