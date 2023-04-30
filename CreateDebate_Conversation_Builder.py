from typing import Dict, Any
from conversant.Conversation import Conversation, Message
import pandas as pd

class DataFrameConversationReader:
    def __init__(self, parse_strategy: Dict[str, str]):
        self.parse_strategy = parse_strategy

    def parse(self, df: pd.DataFrame) -> Conversation:
        messages = []
        for _, row in df.iterrows():
            message_data = {}
            for key, value in self.parse_strategy.items():
                message_data[key] = row[value]
            messages.append(Message(**message_data))
        return Conversation(messages)

    def parse_messages(self, df: pd.DataFrame):
        messages = []
        for _, row in df.iterrows():
            message_data = {}
            for key, value in self.parse_strategy.items():
                message_data[key] = row[value]
            messages.append(Message(**message_data))
        return messages

    def messages2Conversation(self, messages) -> Conversation:
        return Conversation(messages)