
# mongo_logger.py

from pymongo import MongoClient
from datetime import datetime, timezone


class MongoLogger:

    def __init__(self):

        self.client = MongoClient(
            "mongodb://localhost:27017/"
        )

        self.db = self.client["llm_monitoring"]

        self.collection = self.db["call_logs"]


    def log_call(self, log_data):

        log_data["timestamp"] = datetime.now(timezone.utc)

        self.collection.insert_one(log_data)

        print("Call details saved in MongoDB!")


    def close(self):

        self.client.close()
