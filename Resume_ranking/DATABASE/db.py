import os
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Mongo URI from environment
MONGO_URI = os.getenv('MONGO_URI')

# Global variable for client
client = None

def get_database():
    global client
    try:
        if client is None:
            # Create a new MongoClient instance
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 seconds timeout

            # Force a connection attempt
            client.server_info()


        db = client['Resume_ranking']  # Your DB name

        # Test the connection
        db.command('ping')
        return db, client  # return both db and client for transaction support

    except errors.ServerSelectionTimeoutError as err:
        print("MongoDB server selection error:", err)
        raise Exception("Database connection failed. Please check MongoDB server or internet connection.")

    except errors.PyMongoError as err:
        print("General MongoDB error:", err)
        raise Exception("A MongoDB error occurred.")

    except Exception as err:
        print("Unexpected error:", err)
        raise
