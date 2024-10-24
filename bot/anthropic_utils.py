import config
from api_utils import LLM

import base64
from io import BytesIO
import anthropic


ANTHROPIC_MESSAGES_PARAMS = {
    "max_tokens": 1024,
}
ANTHROPIC_AVAILABLE_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-latest",
}
ANTHROPIC_VISION_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-latest",
}


class Claude(LLM):
    def __init__(self, model="claude-3-haiku-20240307"):
        assert model in ANTHROPIC_AVAILABLE_MODELS, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
        )

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._generate_prompt_messages(message, dialog_messages)
                message = messages[-1]["content"]
                prompt = config.chat_modes[chat_mode]["prompt_start"]

                m = client.messages.create(
                    model=self.model,
                    messages=messages,
                    system=prompt,
                    **ANTHROPIC_MESSAGES_PARAMS
                )
                answer = m.content[0].text

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = m.usage.input_tokens, m.usage.output_tokens
            except anthropic.BadRequestError as e:  # too many tokens?
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
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
        )

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

            try:
                messages = self._generate_prompt_messages(message, dialog_messages)
                message = messages[-1]["content"]
                prompt = config.chat_modes[chat_mode]["prompt_start"]

                async with client.messages.stream(
                    model=self.model,
                    messages=messages,
                    system=prompt,
                    **ANTHROPIC_MESSAGES_PARAMS
                ) as stream:
                    answer = ""
                    n_input_tokens = 0
                    n_output_tokens = 0

                    async for event in stream:
                        if event.type == 'message_start':
                            m = event.message
                            n_input_tokens = m.usage.input_tokens
                        elif event.type == 'content_block_delta':
                            answer += event.delta.text
                            yield (
                                "not_finished",
                                answer,
                                (n_input_tokens, n_output_tokens),
                                n_first_dialog_messages_removed,
                                message,
                            )
                        elif event.type == 'message_delta':
                            n_output_tokens = event.usage.output_tokens

                    answer = self._postprocess_answer(answer)

            except anthropic.BadRequestError as e:  # too many tokens?
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
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
        )

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in ANTHROPIC_VISION_MODELS:
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, image_buffer
                    )
                    message = messages[-1]["content"]
                    prompt = config.chat_modes[chat_mode]["prompt_start"]

                    m = client.messages.create(
                        model=self.model,
                        messages=messages,
                        system=prompt,
                        **ANTHROPIC_MESSAGES_PARAMS
                    )
                    answer = m.content[0].text
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = m.usage.input_tokens, m.usage.output_tokens
            except anthropic.BadRequestError as e:  # too many tokens?
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
        client = anthropic.AsyncAnthropic(
            api_key=config.anthropic_api_key,
        )

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

            try:
                if self.model in ANTHROPIC_VISION_MODELS:
                    messages = self._generate_prompt_messages(message, dialog_messages, image_buffer)
                    message = messages[-1]["content"]
                    prompt = config.chat_modes[chat_mode]["prompt_start"]

                    async with client.messages.stream(
                        model=self.model,
                        messages=messages,
                        system=prompt,
                        **ANTHROPIC_MESSAGES_PARAMS
                    ) as stream:
                        answer = ""
                        n_input_tokens = 0
                        n_output_tokens = 0

                        async for event in stream:
                            if event.type == 'message_start':
                                m = event.message
                                n_input_tokens = m.usage.input_tokens
                            elif event.type == 'content_block_delta':
                                answer += event.delta.text
                                yield (
                                    "not_finished",
                                    answer,
                                    (n_input_tokens, n_output_tokens),
                                    n_first_dialog_messages_removed,
                                    message,
                                )
                            elif event.type == 'message_delta':
                                n_output_tokens = event.usage.output_tokens

                        answer = self._postprocess_answer(answer)
            except anthropic.BadRequestError as e:  # too many tokens?
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

    def _generate_prompt_messages(self, message, dialog_messages, image_buffer: BytesIO = None):
        messages = []
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
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": self._encode_image(image_buffer),
                            }
                        }
                    ]
                }
                
            )
        else:
            messages.append({"role": "user", "content": message})

        return messages
    
    def _encode_image(self, image_buffer: BytesIO) -> bytes:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer
