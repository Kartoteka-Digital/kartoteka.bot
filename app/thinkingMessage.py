from aiogram.types import Message

class thinkingMessage:

    def __init__(self, message: Message, text: str = "🤔 Думаю…"):
        self._m = message
        self._text = text
        self._sent = None

    async def __aenter__(self):
        self._sent = await self._m.answer(self._text)
        return self._sent

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        from aiogram.exceptions import TelegramBadRequest
        try:
            await self._m.bot.delete_message(self._m.chat.id, self._sent.message_id)
        except TelegramBadRequest:
            pass
