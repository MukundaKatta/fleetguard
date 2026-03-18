"""Tests for Fleetguard."""
from src.core import Fleetguard
def test_init(): assert Fleetguard().get_stats()["ops"] == 0
def test_op(): c = Fleetguard(); c.track(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Fleetguard(); [c.track() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Fleetguard(); c.track(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Fleetguard(); r = c.track(); assert r["service"] == "fleetguard"
