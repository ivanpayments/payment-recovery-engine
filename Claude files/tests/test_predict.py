from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


SYNTHETIC_PAYLOAD = {
    "amount": 5834.24,
    "amount_usd": 5834.24,
    "currency": "USD",
    "merchant_vertical": "ecom",
    "merchant_mcc": 5734,
    "merchant_country": "US",
    "archetype": "mid_market_saas",
    "processor_name": "global-acquirer-a",
    "routing_reason": "highest_auth",
    "card_brand": "visa",
    "card_type": "credit",
    "card_country": "US",
    "card_funding_source": "consumer",
    "is_cross_border": False,
    "is_token": True,
    "token_type": "network_token",
    "present_mode": "cnp",
    "response_code": "51",
    "response_message": "Insufficient funds",
    "decline_bucket": "soft",
    "is_soft_decline": True,
    "scheme_response_code": "51",
    "three_ds_requested": True,
    "three_ds_outcome": "authenticated",
    "three_ds_version": 2.2,
    "three_ds_flow": "frictionless",
    "three_ds_eci": 5,
    "sca_exemption": "none",
    "latency_ms": 317,
    "latency_auth_ms": 317,
    "latency_3ds_ms": 0,
    "processor_fee_bps": 240,
    "interchange_estimate_bps": 180,
    "scheme_fee_bps": 15,
    "fx_applied": False,
    "fx_rate": 1.0,
    "settlement_currency": "USD",
    "risk_score": 42,
    "fraud_flag": False,
    "risk_model_version": "risk_v3",
    "billing_country": "US",
    "shipping_country": "US",
    "ip_country": "US",
    "issuer_country": "US",
    "payment_method": "card",
    "wallet_type": "none",
    "network_token_present": True,
    "entry_mode": "ecommerce",
    "pan_entry_mode": "manual",
    "cardholder_verification_method": "cvv",
    "cvv_result": "M",
    "avs_result": "Y",
    "avs_zip_match": True,
    "avs_street_match": True,
    "payment_method_details": "visa_credit",
    "issuer_bank_country": "US",
    "card_product_type": "consumer",
    "card_category": "consumer",
    "card_commercial_type": "no",
    "user_agent_family": "mobile_web",
    "timeout_flag": False,
    "device_os": "ios",
    "issuer_size": "large",
    "account_updater_used": False,
    "mastercard_advice_code": 0,
    "routing_optimized": True,
    "mcc_routing_optimized": False,
    "smart_routed": True,
    "scheme_ms": 110,
    "transaction_type": "subscription",
    "fx_bps": 0,
    "routed_network": "visa",
    "risk_skip_flag": False,
    "event_hour": 11,
    "event_dayofweek": 5,
    "event_month": 7,
    "is_weekend": True,
}


class PredictApiTests(unittest.TestCase):
    def test_health(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("decision_threshold", payload)

    def test_predict_fast_path(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict?include_explanation=false", json=SYNTHETIC_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["recommended_action"], {"retry", "do_not_retry"})
        self.assertIn("recovery_probability", payload)
        self.assertNotIn("top_explanation_features", payload)


if __name__ == "__main__":
    unittest.main()
