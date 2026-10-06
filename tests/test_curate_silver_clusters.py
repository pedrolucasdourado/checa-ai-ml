from scripts.curate_silver_clusters import build_curation, curate_cluster


def cluster(**overrides):
    base = {
        "cluster_id": 1,
        "document_count": 100,
        "top_topics": [["golpes_plataformas", 90], ["economia", 10]],
        "top_domains": [["boatos.org", 70], ["g1.globo.com", 30]],
        "top_keywords": [["whatsapp", 80], ["link", 70], ["golpe", 40], ["cadastro", 30]],
        "top_title_terms": [["site", 20], ["grátis", 15]],
        "examples": [],
    }
    return {**base, **overrides}


def test_curate_cluster_labels_whatsapp_scams():
    result = curate_cluster(cluster())

    assert result["suggested_label"] == "golpes_whatsapp_links_cadastros"
    assert result["cluster_type"] == "tema"
    assert result["confidence"] == "alta"


def test_curate_cluster_detects_source_format_dominance():
    result = curate_cluster(
        cluster(
            top_domains=[["g1.globo.com", 90], ["boatos.org", 10]],
            top_keywords=[["whatsapp", 10]],
            top_title_terms=[["fake", 80], ["vídeo", 40]],
        )
    )

    assert result["suggested_label"] == "fato_ou_fake_g1_imagens_videos"
    assert result["cluster_type"] == "fonte_formato"
    assert "fonte/formato" in result["issue"]


def test_build_curation_preserves_metadata_and_flags_draft():
    summary = {
        "metadata": {"documents": 100, "clusters": 1},
        "clusters": [
            cluster(
                top_topics=[["saude_ciencia", 85], ["politica_eleicoes", 15]],
                top_keywords=[["vacina", 50], ["covid", 40]],
                top_title_terms=[["coronavac", 20]],
            )
        ],
    }

    result = build_curation(summary)

    assert result["metadata"]["status"] == "draft_for_human_review"
    assert result["clusters"][0]["suggested_label"] == "vacinas_covid_imunizacao"
