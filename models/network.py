import time
import requests
from typing import Any, Mapping


def wait_for_key(
    url: str,
    key: str,
    expected: Any,
    poll_interval: float = 0.5
) -> bool:
    while True:
        try:
            response: requests.Response = requests.get(url=url)
            response_data: Any = response.json()
            if isinstance(response_data, Mapping) and response_data.get(key) == expected:
                return True

        except Exception:
            pass

        time.sleep(poll_interval)
