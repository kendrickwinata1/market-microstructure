import ctypes
import time

from rest_factory import *


def stars():
    print("*********************************************")


if __name__ == "__main__":
    # demo factory to create REST gateways
    # use the factory to create a sample gateway
    my_restfactory = RestFactory()
    stars()
    print("try futures")
    stars()
    futuretestnet_apikey = ""
    futuretestnet_apisecret = ""
    futuretestnet_base_url = "https://testnet.binancefuture.com"

    futuretestnet_gateway = my_restfactory.create_gateway(
        "BINANCE_TESTNET_FUTURE",
        futuretestnet_base_url,
        futuretestnet_apikey,
        futuretestnet_apisecret,
    )

    print(futuretestnet_gateway.ping())
    print(futuretestnet_gateway.get_price_ticker("BTCUSDT"))
    print(id(futuretestnet_gateway))

    stars()
    print("try spot")
    stars()
    spottestnet_apikey = ""
    spottestnet_apisecret = ""
    spottestnet_base_url = "https://testnet.binance.vision"

    spottestnet_gateway = my_restfactory.create_gateway(
        "BINANCE_TESTNET_SPOT",
        spottestnet_base_url,
        spottestnet_apikey,
        spottestnet_apisecret,
    )
    print(spottestnet_gateway.ping())
    print(spottestnet_gateway.get_price_ticker("BTCUSDT"))
    print(id(spottestnet_gateway))
