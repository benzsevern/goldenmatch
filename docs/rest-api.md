---
layout: default
title: REST API
nav_order: 17
---

# REST API

GoldenMatch includes a local HTTP server for real-time matching, cluster browsing, and data steward review.

---

## Start the server

```bash
goldenmatch serve --file customers.csv --config config.yaml --port 8080
```

The server loads data, runs initial matching, and exposes endpoints on the specified port.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Dataset statistics |
| `POST` | `/match` | Match a single record |
| `POST` | `/match/batch` | Match multiple records |
| `POST` | `/explain` | Explain why two records match |
| `GET` | `/clusters` | List all clusters |
| `GET` | `/clusters/<id>` | Get cluster detail |
| `GET` | `/reviews` | Review queue (borderline pairs) |
| `GET` | `/reviews/decisions` | List completed review decisions |
| `POST` | `/reviews/decide` | Approve or reject a pair |

---

## Health check

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok"}
```

---

## Stats

```bash
curl http://localhost:8080/stats
```

```json
{
    "total_records": 5000,
    "total_clusters": 847,
    "singleton_count": 3200,
    "match_rate": 0.12,
    "avg_cluster_size": 2.1,
    "max_cluster_size": 8
}
```

---

## Match a record

```bash
curl -X POST http://localhost:8080/match \
    -H "Content-Type: application/json" \
    -d '{"name": "John Smith", "zip": "10001"}'
```

```json
{
    "matches": [
        {"row_id": 42, "score": 0.92, "name": "Jon Smyth", "zip": "10001"},
        {"row_id": 108, "score": 0.87, "name": "John Smithe", "zip": "10001"}
    ]
}
```

### Batch match

```bash
curl -X POST http://localhost:8080/match/batch \
    -H "Content-Type: application/json" \
    -d '[
        {"name": "John Smith", "zip": "10001"},
        {"name": "Jane Doe", "zip": "90210"}
    ]'
```

---

## Explain a match

```bash
curl -X POST http://localhost:8080/explain \
    -H "Content-Type: application/json" \
    -d '{"id_a": 42, "id_b": 108}'
```

```json
{
    "overall_score": 0.92,
    "explanation": "Strong match: name similarity 0.88 (jaro_winkler), zip exact match.",
    "field_scores": [
        {"field": "name", "scorer": "jaro_winkler", "score": 0.88},
        {"field": "zip", "scorer": "exact", "score": 1.0}
    ]
}
```

---

## Clusters

### List all clusters

```bash
curl http://localhost:8080/clusters
```

```json
{
    "clusters": [
        {"cluster_id": 1, "size": 3, "confidence": 0.91},
        {"cluster_id": 2, "size": 2, "confidence": 0.87}
    ]
}
```

### Get cluster detail

```bash
curl http://localhost:8080/clusters/1
```

```json
{
    "cluster_id": 1,
    "size": 3,
    "confidence": 0.91,
    "members": [
        {"row_id": 42, "name": "John Smith", "email": "john@example.com"},
        {"row_id": 108, "name": "Jon Smyth", "email": "john@example.com"},
        {"row_id": 215, "name": "J. Smith", "email": "jsmith@example.com"}
    ],
    "pair_scores": {
        "42-108": 0.92,
        "42-215": 0.85,
        "108-215": 0.83
    },
    "bottleneck_pair": [108, 215]
}
```

---

## Review queue

The review queue surfaces borderline pairs for data steward approval or rejection.

### Get pending reviews

```bash
curl http://localhost:8080/reviews
```

```json
{
    "reviews": [
        {
            "id": "rev_001",
            "id_a": 42,
            "id_b": 300,
            "score": 0.82,
            "record_a": {"name": "John Smith", "zip": "10001"},
            "record_b": {"name": "J. Smithson", "zip": "10001"}
        }
    ]
}
```

### Make a decision

```bash
curl -X POST http://localhost:8080/reviews/decide \
    -H "Content-Type: application/json" \
    -d '{"id": "rev_001", "decision": "approve"}'
```

Decision values: `"approve"` or `"reject"`.

### List completed decisions

```bash
curl http://localhost:8080/reviews/decisions
```

---

## Python client

GoldenMatch includes a REST client that uses only stdlib `urllib` (no extra dependencies):

```python
import goldenmatch as gm

client = gm.Client("http://localhost:8080")

# Match a record
matches = client.match({"name": "John Smith", "zip": "10001"})

# List clusters
clusters = client.list_clusters()

# Explain a match
explanation = client.explain(42, 108)

# Get review queue
reviews = client.reviews()
```

---

## Docker deployment

```bash
docker run --rm -p 8080:8080 -v $(pwd):/data \
    ghcr.io/benzsevern/goldenmatch:latest \
    serve --file /data/customers.csv --port 8080
```
