from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    async def send_message(self, message, dialog_messages, chat_mode):
        pass

    @abstractmethod
    async def send_message_stream(self, message, dialog_messages, chat_mode):
        pass
