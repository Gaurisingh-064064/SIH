import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class Block:
    def __init__(
        self,
        index: int,
        timestamp: str,
        data: Dict[str, Any],
        previous_hash: str,
    ):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash

        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }

        encoded_data = json.dumps(
            block_data,
            sort_keys=True,
            default=str,
        ).encode()

        return hashlib.sha256(encoded_data).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class PrivateBlockchain:

    def __init__(self):

        self.chain: List[Block] = []

        self.create_genesis_block()

    def create_genesis_block(self):

        genesis_block = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat(),
            data={
                "action": "GENESIS_BLOCK",
                "message": "NyayaNet Private Blockchain Initialized",
            },
            previous_hash="0",
        )

        self.chain.append(genesis_block)

    def get_latest_block(self) -> Block:

        return self.chain[-1]

    def add_block(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        latest_block = self.get_latest_block()

        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            previous_hash=latest_block.hash,
        )

        self.chain.append(new_block)

        return new_block.to_dict()

    def get_chain(self) -> List[Dict[str, Any]]:

        return [block.to_dict() for block in self.chain]

    def get_block_count(self) -> int:

        return len(self.chain)

    def verify_chain(self) -> bool:

        for index in range(
            1,
            len(self.chain),
        ):

            current_block = self.chain[index]

            previous_block = self.chain[index - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def find_records(
        self,
        investigation_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        records = []

        for block in self.chain:

            data = block.data

            if investigation_id and data.get("investigation_id") != investigation_id:
                continue

            if evidence_id and data.get("evidence_id") != evidence_id:
                continue

            records.append(block.to_dict())

        return records


blockchain = PrivateBlockchain()
