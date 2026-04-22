from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import project3_tool


class Project3ToolTests(unittest.TestCase):
    def test_project3_payload_maps_key_fields(self) -> None:
        row = pd.Series(
            {
                "transaction_id": "txn_123",
                "created_at": "2025-07-26T11:16:06Z",
                "amount": 5834.24,
                "amount_usd": 5834.24,
                "currency": "USD",
                "merchant_vertical": "saas",
                "merchant_mcc": 7372,
                "merchant_country": "US",
                "merchant_tier": "enterprise",
                "processor": "global-acquirer-a",
                "routing_strategy": "highest_auth",
                "card_brand": "visa",
                "card_funding_type": "credit",
                "card_country": "US",
                "is_cross_border": False,
                "is_tokenized": True,
                "token_type": "network_token",
                "presence_mode": "cnp",
                "response_code": "51",
                "response_message": "Insufficient funds",
                "decline_category": "soft",
                "three_ds_version": "2.2.0",
                "three_ds_status": "authenticated",
                "authentication_flow": "frictionless",
                "eci": "05",
                "sca_exemption": "none",
                "processing_time_ms": 317,
                "provider_latency_ms": 110,
                "processing_fee_usd": 53.52,
                "interchange_fee_usd": 40.14,
                "scheme_fee_usd": 2.90,
                "fx_rate": 1.0,
                "settlement_currency": "USD",
                "risk_score": 42,
                "is_fraud": False,
                "risk_provider": "risk_v3",
                "customer_country": "US",
                "customer_ip_country": "US",
                "payment_method_type": "card",
                "payment_method_subtype": "visa_credit",
                "wallet_provider": "",
                "channel": "web",
                "cvv_result": "M",
                "avs_result": "Y",
                "device_type": "desktop",
                "is_outage": False,
                "issuer_parent_group": "large",
                "account_updater_triggered": False,
                "merchant_advice_code": 0,
                "smart_routing": True,
                "transaction_type": "subscription",
                "risk_decision": "review",
            }
        )

        payload = project3_tool.project3_payload_from_transaction(row)
        self.assertEqual(payload["response_code"], "51")
        self.assertTrue(payload["is_soft_decline"])
        self.assertEqual(payload["event_hour"], 11)
        self.assertEqual(payload["processor_name"], "global-acquirer-a")
        self.assertIn("processor_fee_bps", payload)

    @patch("project3_tool.load_transactions")
    def test_call_project3_api_returns_compact_result(self, mock_load: Mock) -> None:
        mock_load.return_value = pd.DataFrame(
            [
                {
                    "transaction_id": "txn_123",
                    "created_at": "2025-07-26T11:16:06Z",
                    "amount": 5834.24,
                    "amount_usd": 5834.24,
                    "currency": "USD",
                    "merchant_vertical": "saas",
                    "merchant_mcc": 7372,
                    "merchant_country": "US",
                    "merchant_tier": "enterprise",
                    "processor": "global-acquirer-a",
                    "routing_strategy": "highest_auth",
                    "card_brand": "visa",
                    "card_funding_type": "credit",
                    "card_country": "US",
                    "is_cross_border": False,
                    "is_tokenized": True,
                    "token_type": "network_token",
                    "presence_mode": "cnp",
                    "response_code": "51",
                    "response_message": "Insufficient funds",
                    "decline_category": "soft",
                    "three_ds_version": "2.2.0",
                    "three_ds_status": "authenticated",
                    "authentication_flow": "frictionless",
                    "eci": "05",
                    "sca_exemption": "none",
                    "processing_time_ms": 317,
                    "provider_latency_ms": 110,
                    "processing_fee_usd": 53.52,
                    "interchange_fee_usd": 40.14,
                    "scheme_fee_usd": 2.90,
                    "fx_rate": 1.0,
                    "settlement_currency": "USD",
                    "risk_score": 42,
                    "is_fraud": False,
                    "risk_provider": "risk_v3",
                    "customer_country": "US",
                    "customer_ip_country": "US",
                    "payment_method_type": "card",
                    "payment_method_subtype": "visa_credit",
                    "wallet_provider": "",
                    "channel": "web",
                    "cvv_result": "M",
                    "avs_result": "Y",
                    "device_type": "desktop",
                    "is_outage": False,
                    "issuer_parent_group": "large",
                    "account_updater_triggered": False,
                    "merchant_advice_code": 0,
                    "smart_routing": True,
                    "transaction_type": "subscription",
                    "risk_decision": "review",
                }
            ]
        )
        fake_httpx = Mock()
        fake_httpx.post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "recovery_probability": 0.31544,
                "recommended_action": "retry",
                "expected_value": 643.97,
                "confidence": 0.53,
                "top_explanation_features": [
                    {"feature": "response_code", "feature_value": "51", "direction": "increase", "business_explanation": "response_code=51 increased the predicted retry recoverability."},
                    {"feature": "is_soft_decline", "feature_value": 1, "direction": "increase", "business_explanation": "is_soft_decline=1 increased the predicted retry recoverability."},
                    {"feature": "amount_usd", "feature_value": 5834.24, "direction": "increase", "business_explanation": "amount_usd=5834.24 increased the predicted retry recoverability."},
                ],
            },
        )

        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = project3_tool.call_project3_api({"transaction_id": "txn_123"})
        self.assertEqual(result["recommended_action"], "retry")
        self.assertIn("response_code=51 increased", result["explanation"])


if __name__ == "__main__":
    unittest.main()
