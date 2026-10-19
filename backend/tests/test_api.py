"""API tests: empty state with no data, and 200s for every endpoint on synthetic fixture data."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.core.state import STATE
from app.main import app
from app.models import autoencoder as ae_mod
from tests.conftest import make_frames, to_bytes

FAST = {"rows": 5, "cols": 5, "ordering_iters": 200, "convergence_iters": 400}


@pytest.fixture
def client(monkeypatch):
    # Never touch the real data/raw/ folder during tests.
    monkeypatch.setattr("app.main.load_from_dir", lambda *a, **k: None)
    with TestClient(app) as c:
        STATE.set_dataset(None)
        yield c
    STATE.set_dataset(None)


def _upload(client):
    feats, pf, vf = make_frames()
    files = {"features": ("f.csv", to_bytes(feats, False)), "prices": ("p.csv", to_bytes(pf, True)),
             "volumes": ("v.csv", to_bytes(vf, True))}
    r = client.post("/api/data/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def _train(client, **extra):
    r = client.post("/api/som/train", json={**FAST, **extra})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    # Consume the SSE stream until the run finishes.
    events = []
    with client.stream("GET", f"/api/som/train/{run_id}/stream") as s:
        for line in s.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events[-1]["type"] in ("done", "error"):
                    break
    assert events[-1]["type"] == "done", events[-1]
    return run_id, events


def test_empty_state(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.json()["loaded"] is False
    for path in ("/api/stocks",):
        r = client.get(path)
        assert r.status_code == 409 and r.json()["detail"]["code"] == "no_data"
    r = client.post("/api/som/train", json=FAST)
    assert r.status_code == 409
    r = client.get("/api/som/abc/map")
    assert r.status_code == 409
    r = client.post("/api/som/step", json={"ticker": "X.NS"})
    assert r.status_code == 409
    r = client.post("/api/lvq/evaluate", json={})
    assert r.status_code == 409


def test_bad_upload_rejected(client):
    files = {"features": ("f.csv", b"a,b\n1,2\n"), "prices": ("p.csv", b"Date,x\n2022-01-03,1\n"),
             "volumes": ("v.csv", b"Date,x\n2022-01-03,1\n")}
    r = client.post("/api/data/upload", files=files)
    assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_data"


def test_full_flow(client):
    status = _upload(client)
    assert status["loaded"] and status["n_stocks"] == 40

    r = client.get("/api/stocks")
    assert r.status_code == 200 and len(r.json()["stocks"]) == 40

    run_id, events = _train(client)
    cps = [e for e in events if e["type"] == "checkpoint"]
    assert len(cps) > 5 and {"eta", "sigma", "qe", "te", "umatrix", "hits", "weights"} <= set(cps[0])

    assert client.get("/api/runs").json()["latest_run_id"] == run_id
    assert client.get("/api/status").json()["latest_run_id"] == run_id

    m = client.get(f"/api/som/{run_id}/map")
    assert m.status_code == 200
    body = m.json()
    assert len(body["umatrix"]) == 5 and len(body["component_planes"]) == 7 and len(body["stocks"]) == 40

    c = client.get(f"/api/som/{run_id}/clusters")
    assert c.status_code == 200 and config.CLUSTER_K_MIN <= c.json()["k"]

    t = body["stocks"][0]["ticker"]
    assert client.get(f"/api/som/{run_id}/stock/{t}").status_code == 200
    assert client.get(f"/api/som/{run_id}/history").status_code == 200

    r1 = client.get(f"/api/som/{run_id}/rq1")
    assert r1.status_code == 200, r1.text
    assert -1 <= r1.json()["ari"] <= 1 and r1.json()["lvq"]["n_samples"] == 40
    assert client.get(f"/api/som/{run_id}/rq2").status_code == 200
    r3 = client.get(f"/api/som/{run_id}/rq3")
    assert r3.status_code == 200, r3.text
    assert r3.json()["n_portfolios"] == config.RQ3_N_PORTFOLIOS

    step = client.post("/api/som/step", json={"ticker": t, "rows": 4, "cols": 4, "eta": 0.5, "sigma": 1.0})
    assert step.status_code == 200
    s = step.json()
    assert s["neighbourhood"][s["bmu"][0]][s["bmu"][1]] == pytest.approx(1.0)
    step2 = client.post("/api/som/step", json={"ticker": t, "rows": 4, "cols": 4, "weights": s["weights_after"]})
    assert step2.status_code == 200

    assert client.post("/api/lvq/evaluate", json={"epochs": 10}).status_code == 200
    assert client.get("/api/som/nope/map").status_code == 404


@pytest.mark.skipif(not ae_mod.AVAILABLE, reason="TensorFlow not installed")
def test_autoencoder_endpoints(client):
    _upload(client)
    r = client.post("/api/autoencoder/train", json={"epochs": 30, "patience": 5})
    assert r.status_code == 200, r.text
    assert len(r.json()["train_loss"]) == len(r.json()["val_loss"]) > 0
    run_id, events = _train(client, input_mode="autoencoder", autoencoder={"epochs": 30, "patience": 5})
    assert any(e["type"] == "autoencoder" for e in events)
    assert client.get(f"/api/som/{run_id}/map").json()["planes_decoded"] is True


def test_autoencoder_disabled_gracefully(client, monkeypatch):
    _upload(client)
    monkeypatch.setattr(ae_mod, "AVAILABLE", False)
    r = client.post("/api/som/train", json={**FAST, "input_mode": "autoencoder"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "ae_unavailable"
    assert client.get("/api/status").json()["autoencoder_available"] is False
