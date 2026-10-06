import json

from scripts.apply_silver_clusters_to_qdrant import (
    CLUSTER_VERSION,
    build_cluster_payload,
    cluster_review_status,
    load_document_payloads,
)


def test_cluster_review_status_approves_only_clear_theme_clusters():
    assert cluster_review_status({"cluster_type": "tema", "confidence": "alta"}) == "approved"
    assert cluster_review_status({"cluster_type": "tema", "confidence": "media"}) == "approved"
    assert cluster_review_status({"cluster_type": "tema", "confidence": "baixa"}) == "needs_subcluster"
    assert cluster_review_status({"cluster_type": "fonte_formato", "confidence": "alta"}) == "needs_subcluster"
    assert cluster_review_status({"cluster_type": "misto", "confidence": "media"}) == "needs_subcluster"


def test_build_cluster_payload_contains_qdrant_fields():
    payload = build_cluster_payload(
        {
            "cluster_id": 2,
            "suggested_label": "golpes_whatsapp_links_cadastros",
            "cluster_type": "tema",
            "confidence": "alta",
        }
    )

    assert payload == {
        "cluster_id": 2,
        "cluster_label": "golpes_whatsapp_links_cadastros",
        "cluster_type": "tema",
        "cluster_confidence": "alta",
        "cluster_version": CLUSTER_VERSION,
        "cluster_review_status": "approved",
    }


def test_load_document_payloads_joins_csv_and_curation(tmp_path):
    curation = {
        "clusters": [
            {
                "cluster_id": 1,
                "suggested_label": "covid_pandemia_saude_publica",
                "cluster_type": "tema",
                "confidence": "media",
            },
            {
                "cluster_id": 9,
                "suggested_label": "fato_ou_fake_g1_imagens_videos",
                "cluster_type": "fonte_formato",
                "confidence": "alta",
            },
        ]
    }
    curation_path = tmp_path / "curation.json"
    curation_path.write_text(json.dumps(curation), encoding="utf-8")
    csv_path = tmp_path / "docs.csv"
    csv_path.write_text(
        "document_id,cluster_id,title\n"
        "doc-a,1,Titulo A\n"
        "doc-b,9,Titulo B\n",
        encoding="utf-8",
    )

    payloads = load_document_payloads(csv_path, curation_path)

    assert payloads["doc-a"]["cluster_review_status"] == "approved"
    assert payloads["doc-b"]["cluster_review_status"] == "needs_subcluster"
