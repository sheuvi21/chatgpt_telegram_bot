import base64
from io import BytesIO
import config
from api_utils import LLM
import logging

import tiktoken
import openai


# setup openai
openai.api_key = config.openai_api_key
if config.openai_api_base is not None:
    openai.api_base = config.openai_api_base
logger = logging.getLogger(__name__)


OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 60.0,
}
OPENAI_AVAILABLE_MODELS = {
    "text-davinci-003",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "o1",
    "o3-mini",
    "gpt-4.1",
    "o3",
    "o4-mini",
}
OPENAI_CHAT_MODELS = {
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "o1",
    "o3-mini",
    "gpt-4.1",
    "o3",
    "o4-mini",
}
OPENAI_VISION_MODELS = {
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4o",
    "o1",
    "gpt-4.1",
    "o3",
    "o4-mini",
}
OPENAI_REASONING_MODELS = {
    "o1-preview",
    "o1-mini",
    "o1",
    "o3-mini",
    "o3",
    "o4-mini",
}


class ChatGPT(LLM):
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in OPENAI_AVAILABLE_MODELS, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        prompt = self._get_system_prompt(chat_mode)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in OPENAI_CHAT_MODELS:
                    messages = self._generate_prompt_messages(message, dialog_messages, prompt=prompt)
                    message = messages[-1]["content"]

                    r = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        **self._get_completion_options()
                    )
                    answer = r.choices[0].message["content"]
                elif self.model == "text-davinci-003":
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=prompt,
                        **self._get_completion_options()
                    )
                    answer = r.choices[0].text
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
            message,
        )

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        prompt = self._get_system_prompt(chat_mode)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in OPENAI_CHAT_MODELS:
                    messages = self._generate_prompt_messages(message, dialog_messages, prompt=prompt)
                    message = messages[-1]["content"]

                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **self._get_completion_options()
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta

                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = 0

                            yield (
                                "not_finished",
                                answer,
                                (n_input_tokens, n_output_tokens),
                                n_first_dialog_messages_removed,
                                message,
                            )
                            

                elif self.model == "text-davinci-003":
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r_gen = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=prompt,
                        stream=True,
                        **self._get_completion_options()
                    )

                    answer = ""
                    async for r_item in r_gen:
                        answer += r_item.choices[0].text
                        n_input_tokens, n_output_tokens = self._count_tokens_from_prompt(prompt, answer, model=self.model)
                        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                        yield (
                            "not_finished",
                            answer,
                            (n_input_tokens, n_output_tokens),
                            n_first_dialog_messages_removed,
                            message,
                        )

                answer = self._postprocess_answer(answer)

            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield (
            "finished",
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
            message,
        )  # sending final answer

    async def send_vision_message(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        prompt = self._get_system_prompt(chat_mode)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in OPENAI_VISION_MODELS:
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, prompt=prompt, image_buffer=image_buffer
                    )
                    message = messages[-1]["content"]
                    r = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        **self._get_completion_options()
                    )
                    answer = r.choices[0].message.content
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = (
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens,
                )
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(
            dialog_messages
        )

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
            message,
        )

    async def send_vision_message_stream(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        prompt = self._get_system_prompt(chat_mode)

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in OPENAI_VISION_MODELS:
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, prompt=prompt, image_buffer=image_buffer
                    )
                    message = messages[-1]["content"]
                    
                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **self._get_completion_options(),
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            (
                                n_input_tokens,
                                n_output_tokens,
                            ) = self._count_tokens_from_messages(
                                messages, answer, model=self.model
                            )
                            n_first_dialog_messages_removed = (
                                n_dialog_messages_before - len(dialog_messages)
                            )
                            yield (
                                "not_finished",
                                answer,
                                (n_input_tokens, n_output_tokens),
                                n_first_dialog_messages_removed,
                                message,
                            )

                answer = self._postprocess_answer(answer)

            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield (
            "finished",
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
            message,
        )

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> bytes:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _generate_prompt_messages(
            self,
            message,
            dialog_messages,
            prompt=None,
            image_buffer: BytesIO = None
        ):
        messages = []

        if prompt is not None:
            messages.append({"role": "system", "content": prompt})
        
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
                    
        if image_buffer is not None:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": message,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._encode_image(image_buffer)}",
                            }
                        }
                    ]
                }
                
            )
        else:
            messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        if model in ("gpt-4.1", "o4-mini"):
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model in OPENAI_CHAT_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            if isinstance(message["content"], list):
                for sub_message in message["content"]:
                    if "type" in sub_message:
                        if sub_message["type"] == "text":
                            n_input_tokens += len(encoding.encode(sub_message["text"]))
                        elif sub_message["type"] == "image_url":
                            pass
            else:
                if "type" in message:
                    if message["type"] == "text":
                        n_input_tokens += len(encoding.encode(message["text"]))
                    elif message["type"] == "image_url":
                        pass


        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens
    
    def _get_system_prompt(self, chat_mode):
        if self.model in OPENAI_REASONING_MODELS:
            return None

        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        return config.chat_modes[chat_mode]["prompt_start"]

    def _get_completion_options(self):
        if self.model in OPENAI_REASONING_MODELS:
            return {}
        
        return OPENAI_COMPLETION_OPTIONS


async def transcribe_audio(audio_file) -> str:
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"] or ""


async def generate_images(prompt, n_images=4, size="1024x1024"):
    r = await openai.Image.acreate(prompt=prompt, n=n_images, size=size, model="dall-e-3")
    image_urls = [item.url for item in r.data]
    return image_urls


async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())