from rest_connect.rest import *


class RestFactory:
    """
    Rest Factory class

    This class is used to generate further gateway objects which impelements an abstract gateway interface
    """

    def __init__(self) -> None:
        self._gateway_count = 0
        print("initializing REST gateway factory")

    def get_gateway_count(self) -> int:
        return self._gateway_count

    def create_gateway(
        self,
        gateway_type: str,
        base_url: str,
        api_key: str,
        api_secret: str,
    ) -> BaseRESTGateway:
        """_summary_

        Args:
            str (gateway_type): type of gateway you want to create
            str (base_url): base url, refer to the API docu
            str (api_key): your API key
            str (api_secret): your API secret key

        Returns:
            BaseRESTGateway: gateway object
        """

        if gateway_type == "BINANCE_TESTNET_SPOT":
            self._gateway_count = self._gateway_count + 1
            return SpotTestnetGateway(base_url, api_key, api_secret)
        elif gateway_type == "BINANCE_TESTNET_FUTURE":
            self._gateway_count = self._gateway_count + 1
            return FutureTestnetGateway(base_url, api_key, api_secret)
        else:
            print("unknown gateway type")
            return None
