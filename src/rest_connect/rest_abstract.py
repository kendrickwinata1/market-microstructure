from abc import ABC, abstractmethod


class AbstractRESTGateway(ABC):
    @abstractmethod
    def ping(self):
        pass

    @abstractmethod
    def time(self):
        pass

    @abstractmethod
    def get_price_ticker(self):
        pass

    @abstractmethod
    def get_all_orders(self):
        pass

    @abstractmethod
    def get_account_balance(self):
        pass

    @abstractmethod
    def get_position_info(self):
        pass

    @abstractmethod
    def send_order(self):
        pass

    @abstractmethod
    def cancel_order(self):
        pass

    @abstractmethod
    def cancel_all_order(self):
        pass

    @abstractmethod
    def modify_order(self):
        pass
    
    @abstractmethod
    def get_all_open_orders(self):
        pass