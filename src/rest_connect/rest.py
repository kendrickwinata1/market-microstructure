import hashlib
import hmac
import time
from urllib.parse import urlencode

import requests
from rest_connect.rest_abstract import AbstractRESTGateway


def create_query(base_url, api_url, my_api_key, my_api_secret, order_params):
    # create query string
    query_string = urlencode(order_params)
    # signature
    signature = hmac.new(
        my_api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # print(my_api_secret)
    url = base_url + api_url + "?" + query_string + "&signature=" + signature
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json;charset=utf-8",
            "X-MBX-APIKEY": my_api_key,
        }
    )
    response = session.get(url=url, params={})
    response_data = response.json()

    return response_data


def create_delete(base_url, api_url, my_api_key, my_api_secret, order_params):
    # create query string
    query_string = urlencode(order_params)
    # signature
    signature = hmac.new(
        my_api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # print(my_api_secret)
    url = base_url + api_url + "?" + query_string + "&signature=" + signature
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json;charset=utf-8",
            "X-MBX-APIKEY": my_api_key,
        }
    )
    response = session.delete(url=url, params={})
    response_data = response.json()

    return response_data


class BaseRESTGateway(AbstractRESTGateway):

    # parameterized constructor
    def __init__(self, base_url, api_key, api_secret):
        self._base_url = base_url
        self._api_key = api_key
        self._api_secret = api_secret


class FutureTestnetGateway(BaseRESTGateway):
    # test connection
    def ping(self):
        api_url = "/fapi/v1/ping"
        url = self._base_url + api_url
        response = requests.get(url=url)
        response_data = response.json()
        print("Response: {}".format(response_data))
        return response_data

    def time(self):
        api_url = "/fapi/v1/time"
        url = self._base_url + api_url
        response = requests.get(url=url)
        response_data = response.json()
        print("Response: {}".format(response_data))
        return response_data

    def get_price_ticker(self, symbol):
        api_url = "/fapi/v2/ticker/price"
        # market order parameters
        order_params = {
            "symbol": symbol,
        }
        # create query string
        query_string = urlencode(order_params)
        # signature
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        # print(my_api_secret)
        url = self._base_url + api_url + "?" + query_string
        session = requests.Session()
        response = session.get(url=url, params={})
        response_data = response.json()
        return response_data

    def get_all_orders(self, symbol, timestamp):
        # not sure about the kwargs yet
        api_url = "/fapi/v1/allOrders"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}
        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def get_all_open_orders(self, symbol, timestamp):
        # not sure about the kwargs yet
        api_url = "/fapi/v1/openOrders"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}
        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def get_account_balance(self, timestamp):
        api_url = "/fapi/v2/balance"
        # market order parameters
        order_params = {
            "timestamp": timestamp,
        }
        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def get_position_info(self, symbol, timestamp):
        api_url = "/fapi/v2/positionRisk"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}

        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def send_order(self):
        pass

    def cancel_order(self, symbol, timestamp, orderid):
        api_url = "/fapi/v1/order"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp, "orderid": orderid}
        # create the query
        return create_delete(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def cancel_all_order(self, symbol, timestamp):
        api_url = "/fapi/v1/allOpenOrders"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}
        # create the query
        return create_delete(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def modify_order(self):
        pass


class SpotTestnetGateway(BaseRESTGateway):
    # test connection
    def ping(self):
        api_url = "/api/v3/ping"
        url = self._base_url + api_url
        response = requests.get(url=url)
        response_data = response.json()
        print("Response: {}".format(response_data))

    def time(self):
        api_url = "/fapi/v1/time"
        url = self._base_url + api_url
        response = requests.get(url=url)
        response_data = response.json()
        print("Response: {}".format(response_data))

    def get_price_ticker(self, symbol):
        api_url = "/api/v3/ticker/price"
        # market order parameters
        order_params = {
            "symbol": symbol,
        }
        # create query string
        query_string = urlencode(order_params)
        # print(my_api_secret)
        url = self._base_url + api_url + "?" + query_string
        session = requests.Session()
        response = session.get(url=url, params={})
        response_data = response.json()
        return response_data

    def get_all_orders(self, symbol: str, timestamp: int):
        # not sure about the kwargs yet
        api_url = "/api/v3/allOrders"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}
        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )

    def get_account_balance(self):
        # only /sapi endpoint available,
        # spot testnet can only use /api endpoint
        # refer to https://testnet.binance.vision/
        pass

    def get_position_info(self, timestamp: int):
        # only /sapi endpoint available,
        # spot testnet can only use /api endpoint
        # refer to https://testnet.binance.vision/
        pass

    def send_order(self):
        pass

    def cancel_order(self):
        pass

    def cancel_all_order(self):
        pass

    def modify_order(self):
        pass

    def get_all_open_orders(self, symbol, timestamp):
        # not sure about the kwargs yet
        api_url = "/fapi/v1/openOrders"
        # market order parameters
        order_params = {"symbol": symbol, "timestamp": timestamp}
        # create the query
        return create_query(
            self._base_url, api_url, self._api_key, self._api_secret, order_params
        )
