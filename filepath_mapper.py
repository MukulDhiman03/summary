import json
import os

class FilePathMapper:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.db_dir = "db"
        self.raw_db_dir = "raw_db"
        self.json_db_dir = "json_db"
        self.faiss_db_dir = "faiss_db"
        self.ppt_db_dir = "ppt_db"


        # Construct full paths
        self.faiss_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.faiss_db_dir)
        self.raw_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.raw_db_dir)
        self.json_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.json_db_dir)
        self.ppt_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.ppt_db_dir)

        # Create required directories if they don't exist
        for dir_path in [self.faiss_db_dir_path, self.json_db_dir_path]:
            os.makedirs(dir_path, exist_ok=True)

    def get_faiss_json_file_paths(self, file_path: str):
        """Given a raw file path, return corresponding JSON and FAISS file paths."""
        if self.raw_db_dir not in file_path:
            raise ValueError(f"Expected '{self.raw_db_dir}' in file path, got: {file_path}")

        _, post_file_path = file_path.split(self.raw_db_dir, 1)
        post_file_path = post_file_path.lstrip("/")

        post_file_dir, filename = os.path.split(post_file_path)
        filename_wo_ext, _ = os.path.splitext(filename)

        json_post_file_path = os.path.join(post_file_dir, filename_wo_ext + ".json")
        faiss_post_file_path = os.path.join(post_file_dir, filename_wo_ext + ".faiss")

        json_file_path = os.path.join(self.json_db_dir_path, json_post_file_path)
        faiss_file_path = os.path.join(self.faiss_db_dir_path, faiss_post_file_path)

        return json_file_path, faiss_file_path
