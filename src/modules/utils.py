from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder.

    method 1: based on the root dir name
    
    ```python
    root_dir_name = 'RAG'
    for p in current_dir.parents:
    if p.name.lower() == root_dir_name.lower():
        root_dir = p
        break
    else:
        raise Exception(f"Root dir \"{root_dir_name}\" Not found")
    ```
    """
    current_dir = Path(os.getcwd())

    # method 2: based on the ".git" dir presence
    for _ in current_dir.parents:
        if ".git" in os.listdir(current_dir.parent) or ".project-root" in os.listdir(current_dir.parent):
            root_dir = current_dir.parent
            print(f"Root Directory: {str(root_dir)}")
            return root_dir
    else:
        raise Exception("No root directory was found that contains a .git directory")


class AtlasClient:
    """A class to connect to MongoDB Atlas and create a vector search index.

    Example Usage:
    ```python
    client = AtlasClient(dbname="VMD_RAG", collection_name="VMD_PDF")
    client.ping() # test connection
    client.create_indexes(embedding_model, index_name="vmd_pdf_index") # create a vector search index
    client.init_vector_store(embedding_model) # initialize the vector store object
    vector_store = client.vector_store # get the vector store object
    ids = client.vector_store.add_documents(documents=all_splits) # add documents to the vector store
    ```
    """

    def __init__(self, atlas_uri=None, dbname: str = None, collection_name: str = None, index_name: str = None):
        if atlas_uri is None:
            atlas_uri = os.getenv("MONGODB_URI")
        if atlas_uri is None:
            raise ValueError("Please provide a valid MongoDB Atlas URI or set MONGODB_URI in .env file")

        self._clt = MongoClient(atlas_uri)
        self.database = self._clt[dbname] if dbname is not None else None
        self.collection = self.database[collection_name] if self.database is not None else None
        self.index_name = index_name

        self._init_vector_store = False  # used in vector_store, to be used with vector_store property.
        self._similarity = None  # used in init_vector_store() and create_indexes()

    # A quick way to test if we can connect to Atlas instance
    def ping(self, debug=True):
        try:
            self._clt.admin.command("ping")
            print("Ping: Successfully connected to MongoDB!") if debug else None
        except Exception as e:
            print("Ping:", e)
            print(
                "You may need to add your IP address to Network Access list in MongoDB deployment\n"
                "https://cloud.mongodb.com -> Security -> Network Access"
            ) if debug else None

    def get_indexes(self) -> list:
        return [d["name"] for d in list(self.collection.list_search_indexes())]
    
    def create_indexes(self, embedding_model, index_name: str, index_def: dict = None):
        existing_indexes = self.get_indexes()
        if index_name in existing_indexes:
            print(f"Index {index_name} already exists.")
            return

        # get the embedding length
        embedding_length = len(embedding_model.embed_query("test"))
        # Define the vector search index
        vector_search_index_definition = (
            {"fields": [{"type": "vector", "path": "embedding", "similarity": "cosine", "numDimensions": embedding_length}]}
            if index_def is None
            else index_def
        )
        # Create the search index model
        search_index_model = SearchIndexModel(definition=vector_search_index_definition, name=index_name, type="vectorSearch")
        # Create the index on the collection
        self.collection.create_search_index(model=search_index_model)
        self.index_name = index_name
        self._similarity = vector_search_index_definition["fields"][0]["similarity"]

    def init_vector_store(self, embedding_model, score_fn: str = None):
        if self.collection is None or self.index_name is None:
            raise ValueError("Run reinit(...) with db, collection, and index names as needed.")

        if score_fn is None and self._similarity is not None:
            score_fn = self._similarity
        elif score_fn is None:
            score_fn = "cosine"

        print(f"Using similarity function: {score_fn} and index: {self.index_name}")

        self._vector_store = MongoDBAtlasVectorSearch(
            embedding=embedding_model, collection=self.collection, index_name=self.index_name, relevance_score_fn=score_fn
        )
        self._init_vector_store = True

    @property
    def vector_store(self):
        if not self._init_vector_store:
            raise ValueError("Please run init_vector_store(...) first.")
        return self._vector_store

    # init a new collection
    def reinit(self, dbname: str = None, collection_name: str = None, index_name: str = None):
        self.database = self._clt[dbname] if dbname is not None else self.database
        self.collection = self.database[collection_name] if collection_name is not None else self.collection
        self.index_name = index_name if index_name is not None else self.index_name


if __name__ == "__main__":
    pass