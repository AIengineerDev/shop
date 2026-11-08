from __future__ import annotations

import time

import pandas as pd
from fastapi import FastAPI

from shopfeed.schema import EventsIngest, EventsIngestResponse, FeedRequest, FeedResponse
from shopfeed.serving.pipeline import FeedService


def create_app(service: FeedService) -> FastAPI:
    app = FastAPI(title="Shop feed", version="0.1.0")
    app.state.service = service

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": service.model_version, "impressions_logged": len(service.logger)}

    @app.post("/feed", response_model=FeedResponse)
    def feed(req: FeedRequest):
        resp = service.recommend(req)
        now = req.now if req.now is not None else time.time()
        orders = service.in_flight_orders(req.user_id, now)
        if orders:
            resp.timings_ms["order_status_cards"] = float(len(orders))
        return resp

    @app.post("/events", response_model=EventsIngestResponse)
    def events(body: EventsIngest):
        df = pd.DataFrame([e.model_dump() for e in body.events])
        res = service.ingest_events(df, now=time.time() + 1e9)  # API trusts client clocks only up to far future guard
        return EventsIngestResponse(**res)

    return app
