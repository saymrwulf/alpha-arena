"""Polygon wallet integration for USDC management."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from eth_account import Account
from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

# Polygon mainnet USDC contract
USDC_CONTRACT_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_DECIMALS = 6

# Polymarket CTF Exchange contract
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Standard ERC20 ABI for balance/transfer
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]


@dataclass
class WalletInfo:
    """Wallet state information."""

    address: str
    usdc_balance: Decimal
    matic_balance: Decimal
    usdc_allowance: Decimal  # Allowance to CTF Exchange


class PolygonWallet:
    """Manage wallet on Polygon network."""

    def __init__(
        self,
        private_key: str,
        rpc_url: str = "https://polygon-rpc.com",
    ):
        self.private_key = private_key
        self.rpc_url = rpc_url
        self._web3: AsyncWeb3 | None = None
        self._account: Account | None = None
        self._usdc_contract: Any = None

    @property
    def address(self) -> str:
        """Get wallet address."""
        if self._account is None:
            self._account = Account.from_key(self.private_key)
        return self._account.address

    async def connect(self) -> None:
        """Connect to Polygon RPC."""
        self._web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.rpc_url))
        self._web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self._account = Account.from_key(self.private_key)

        self._usdc_contract = self._web3.eth.contract(
            address=self._web3.to_checksum_address(USDC_CONTRACT_ADDRESS),
            abi=ERC20_ABI,
        )

    async def disconnect(self) -> None:
        """Disconnect from Polygon."""
        self._web3 = None
        self._usdc_contract = None

    async def get_info(self) -> WalletInfo:
        """Get current wallet state."""
        if self._web3 is None or self._usdc_contract is None:
            raise RuntimeError("Wallet not connected")

        address = self._web3.to_checksum_address(self.address)

        # Get MATIC balance
        matic_wei = await self._web3.eth.get_balance(address)
        matic_balance = Decimal(str(matic_wei)) / Decimal("1e18")

        # Get USDC balance
        usdc_raw = await self._usdc_contract.functions.balanceOf(address).call()
        usdc_balance = Decimal(str(usdc_raw)) / Decimal(f"1e{USDC_DECIMALS}")

        # Get USDC allowance to CTF Exchange
        ctf_address = self._web3.to_checksum_address(CTF_EXCHANGE_ADDRESS)
        allowance_raw = await self._usdc_contract.functions.allowance(
            address, ctf_address
        ).call()
        usdc_allowance = Decimal(str(allowance_raw)) / Decimal(f"1e{USDC_DECIMALS}")

        return WalletInfo(
            address=address,
            usdc_balance=usdc_balance,
            matic_balance=matic_balance,
            usdc_allowance=usdc_allowance,
        )

    async def get_usdc_balance(self) -> Decimal:
        """Get USDC balance."""
        info = await self.get_info()
        return info.usdc_balance

    async def ensure_allowance(self, amount: Decimal) -> str | None:
        """Ensure USDC allowance is sufficient, approve if needed.

        Returns transaction hash if approval was needed, None otherwise.
        """
        if self._web3 is None or self._usdc_contract is None or self._account is None:
            raise RuntimeError("Wallet not connected")

        info = await self.get_info()
        if info.usdc_allowance >= amount:
            return None

        # Need to approve
        amount_raw = int(amount * Decimal(f"1e{USDC_DECIMALS}"))
        ctf_address = self._web3.to_checksum_address(CTF_EXCHANGE_ADDRESS)

        # Build approval transaction
        nonce = await self._web3.eth.get_transaction_count(self.address)
        gas_price = await self._web3.eth.gas_price

        tx = await self._usdc_contract.functions.approve(
            ctf_address, amount_raw
        ).build_transaction({
            "from": self.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": 100000,
        })

        # Sign and send
        signed = self._account.sign_transaction(tx)
        tx_hash = await self._web3.eth.send_raw_transaction(signed.raw_transaction)

        # Wait for confirmation
        await self._web3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_hash.hex()

    async def check_sufficient_balance(self, amount: Decimal) -> bool:
        """Check if wallet has sufficient USDC for a trade."""
        balance = await self.get_usdc_balance()
        return balance >= amount

    async def check_gas_available(self, min_matic: Decimal = Decimal("0.01")) -> bool:
        """Check if wallet has enough MATIC for gas."""
        info = await self.get_info()
        return info.matic_balance >= min_matic
