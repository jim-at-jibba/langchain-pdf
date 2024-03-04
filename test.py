from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token)


# Controls how OpenAI responds to Langchain
# This will make OpenAI to stream to Langchain but we
# wont see anything as Langchain does not stream becasue of this flag
chat = ChatOpenAI(streaming=True, callbacks=[StreamingHandler()])

prompt = ChatPromptTemplate.from_messages([("human", "{content}")])

# LLMChain will not stream the response, even with the stream method
# You must override the stream method if you want streaming
chain = LLMChain(llm=chat, prompt=prompt)

output = chain.stream(input={"content": "tell me a joke"})
print(output)

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
