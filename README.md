# Marketing Analytics (Personal Project)

## About

Marketing measurement on synthetic data with known truth: media mix modeling R2 0.7146, geo incrementality true 0.15 estimated 0.1491, position-based attribution, and customer segmentation. Not affiliated with Wealthfront.

A marketing measurement app built to cover the incrementality
and media mix modeling gap in the Wealthfront Data Scientist,
Marketing role: campaign performance, paths to conversion,
incrementality estimation, media mix modeling, client
segmentation, AB experiment design, and client revenue modeling.

Not affiliated with Wealthfront. No Wealthfront data, no real
campaign data, no real customers.

## About

Media mix modeling with adstock and saturation, geo
incrementality experiment with known ground truth, position-
based attribution, and customer segmentation, on synthetic
data, with a bring-your-own-data upload.

## Data: synthetic, known ground truth

104 weeks, 5 channels (TV, Search, Social, Email, Display),
about $9.92M spend, 230,277 conversions, $27.63M revenue.
Geo experiment: 50 treated geos vs 50 control geos, with a
KNOWN true lift of 15% in the campaign period.
User journeys: 5,000 synthetic journeys, conversion rate
0.2336, with channel-specific conversion weights (Search
highest, then Email), so attribution reflects channel value.

Limit, stated plainly: synthetic data validates the methods
against known truth, it does not prove performance on real
campaigns. A first-order Markov removal effect was tried and
removed, not hidden: it gave 0.0 for every channel, because
conversion here is assigned at the journey level, not per
touch, so a Markov walk ends at conversion with almost the
same probability no matter which channel is removed.
Position-based attribution is reported instead.

## Verified results (local run, Nov 3, 2026)

- `verify.py`: PASS, including MMM R2, incrementality within
  5 points of truth, attribution shares summing to 1, and
  segmentation sizes summing to 2,000
- Media mix modeling: R2 0.7146, contribution shares Search
  0.2794, Email 0.2600, TV 0.2208, Display 0.1464, Social 0.0935
- Incrementality: true lift 0.1500, difference-in-differences
  estimate 0.1491, 50 treated vs 50 control geos, 20 pre and
  20 post weeks, 150.26 conversions per geo
- Attribution, position-based: Search 0.2445, Email 0.2202,
  TV 0.1964, Social 0.1702, Display 0.1686
- Segmentation: 2,000 customers, 4 KMeans segments, sizes
  485, 509, 506, 500, with mean recency, frequency, and money
  per segment in metrics.json
- Live server: health ok, incrementality endpoint matches
  above, home page HTTP 200

## Run it

```
pip install -r requirements.txt
python verify.py
python train.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

- `GET /api/health`, `GET /api/metrics`, `GET /api/mmm`
- `GET /api/incrementality`, `GET /api/attribution`, `GET /api/segments`
- `POST /api/upload` (weekly CSV with spend by channel and
  conversions, MMM run in memory, nothing stored)

## Files

- `train.py`: synthetic generation, MMM, incrementality,
  attribution, segmentation, evaluation
- `app.py`: FastAPI server, BYOD upload
- `verify.py`: end-to-end verification against known truth
- `frontend/index.html`: dashboard, no build step
- `metrics.json`: metrics from the last verified run
