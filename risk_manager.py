# risk_manager.py
import logging

class RiskManager:
    def __init__(self, config):
        self.config = config

    def check(self, inventory, pnl):
        # Inventory position limit check
        if abs(inventory) > self.config.position_limit:
            logging.warning("RiskManager: Inventory limit exceeded.")
            return False
        # Drawdown control
        if pnl < -self.config.max_drawdown:
            logging.warning("RiskManager: Max drawdown breached.")
            return False
        return True
