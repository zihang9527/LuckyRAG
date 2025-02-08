from collections import defaultdict
from typing import List, Any, Optional, Dict

class BaseParser:
    """
    Top class of data parser
    """
    type = None
    def __init__(self) -> None:
        pass
        # self._metadata: Optional[defaultdict] = None
        # self.parse_output: Any = None

    def parse(self):
        raise NotImplementedError()