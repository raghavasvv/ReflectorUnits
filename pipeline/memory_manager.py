import os
import json

class MemoryManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_file_path(self, agent_id):
        return os.path.join(self.base_dir, f"{agent_id}.json")

    def load(self, agent_id):
        path = self._get_file_path(agent_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def save(self, agent_id, memory):
        path = self._get_file_path(agent_id)
        with open(path, "w") as f:
            json.dump(memory, f, indent=2)

    def append(self, agent_id, new_entry):
        memory = self.load(agent_id)
        memory.append(new_entry)
        self.save(agent_id, memory)

    def reset(self, agent_id):
        """Clear memory for a given agent before a new run"""
        path = self._get_file_path(agent_id)
        if os.path.exists(path):
            os.remove(path)
