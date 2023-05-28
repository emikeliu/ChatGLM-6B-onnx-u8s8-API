import uvicorn
from model import ChatGLMModel, chat_template
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List, Tuple, Union
from enum import Enum
class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class Dialog(BaseModel):
    role: Role
    content: str

class HyperArguments(BaseModel):
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0
    max_generated_tokens: int = 2048


class Context(BaseModel):
    messages: List[Dialog]
    config: Union[HyperArguments, None]

class PreprocessedContext(BaseModel):
    system: str
    dialog: List[Tuple[str,str]]
    prompt: str

class Tokens(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Message(BaseModel):
    role: str = 'assistant'
    context: str = ''

class Choices(BaseModel):
    message: Message
    finish_reason: str
    index: int

class Resposes(BaseModel):
    id: str = 'chatcmpl-id'
    object: str = 'chat.completion'
    created: int = 1677649420
    model: str = 'chatglm'
    usage: Tokens
    choices: List[Choices]

app = FastAPI()
model = ChatGLMModel()

def preprocess_context(context: Context) -> PreprocessedContext:
    ret = PreprocessedContext(system="",dialog=[],prompt="")
    current_dialog=["",""]
    cnt = 0
    for dialog in context.messages:
        if(dialog.role == Role.system):
            ret.system = dialog.content
        elif(dialog.role == Role.user):
            current_dialog[0] = dialog.content
            cnt = cnt + 1
        elif(dialog.role == Role.assistant):
            current_dialog[1] = dialog.content
            cnt = cnt + 1
        if(cnt % 2 == 0 and cnt != 0):
            ret.dialog.append(tuple(current_dialog))
            current_dialog = ["",""]
    ret.prompt = current_dialog[0]
    return ret
@app.post("/chat")
async def chat(context: Context) :
    ret = ""
    responses = {
        'id':'chatcmpl-id',
        'object':'chat.completion',
        'created':1677649420,
        'model':'chatglm',
        'usage': {
                'prompt_tokens':56,
                'completion_tokens':31,
                'total_tokens':87
            },
        'choices': [
            {
                'message': {
                    'role':'assistant',
                    'content':''
                },
                'finish_reason':'stop',
                'index':0
            }
        ]
    }
    data = preprocess_context(context)
    if(context.config is None):
        context.config = HyperArguments(top_k=50,top_p=1.0,temperature=1.0,max_generated_tokens=2048)
    prompt = chat_template(data.dialog, data.prompt, data.system)
    for answer in model.generate_iterate(
        prompt,
        max_generated_tokens=context.config.max_generated_tokens,
        top_k=context.config.top_k,
        top_p=context.config.top_p,
        temperature=context.config.temperature,
    ):
        ret = answer
    responses["choices"][0]["message"]["content"] = ret
    return responses

if __name__  == '__main__':
    ChatGLMModel()
    uvicorn.run(app, host="0.0.0.0", port=7860)