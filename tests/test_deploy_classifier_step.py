"""O passo do classificador no deploy degrada de forma explícita e nunca bloqueia sem STRICT."""
import os
import pathlib
import subprocess

import pytest

yaml = pytest.importorskip("yaml")

DEPLOY = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


@pytest.fixture(scope="module")
def step_script(tmp_path_factory):
    steps = yaml.safe_load(DEPLOY.read_text())["jobs"]["deploy"]["steps"]
    run = next(s for s in steps if s.get("id") == "classifier")["run"]
    path = tmp_path_factory.mktemp("step") / "classifier.sh"
    path.write_text(run)
    return path


def execute(step_script, tmp_path, *, want, strict, pointer=False, pip_fails=False):
    (tmp_path / "bin").mkdir()
    (tmp_path / "models").mkdir()
    if pointer:
        (tmp_path / "models" / "best_model_transformer.dvc").write_text("ok")
    if pip_fails:
        pip = tmp_path / "bin" / "pip"
        pip.write_text("#!/bin/sh\nexit 1\n")
        pip.chmod(0o755)
    output, summary = tmp_path / "output", tmp_path / "summary"
    output.write_text("")
    summary.write_text("")
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "WANT": want, "STRICT": strict, "HOST": "h", "VM_USER": "u", "MODELS_DIR": "/x",
        "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = subprocess.run(["bash", "-e", str(step_script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    return result, output.read_text().strip(), summary.read_text()


def test_disabled_does_nothing(step_script, tmp_path):
    result, output, _ = execute(step_script, tmp_path, want="false", strict="false")
    assert result.returncode == 0
    assert output == "enabled=false"
    assert "::warning::" not in result.stdout


def test_enabled_without_pointer_degrades_loudly_and_does_not_block(step_script, tmp_path):
    result, output, summary = execute(step_script, tmp_path, want="true", strict="false")
    assert result.returncode == 0
    assert output == "enabled=false"
    assert "::warning::" in result.stdout
    assert "desligado" in summary


def test_strict_mode_fails_the_deploy(step_script, tmp_path):
    result, output, _ = execute(step_script, tmp_path, want="true", strict="true")
    assert result.returncode == 1
    assert "::error::" in result.stdout
    assert output == ""


def test_error_midway_is_detected_not_ignored(step_script, tmp_path):
    result, output, _ = execute(step_script, tmp_path, want="true", strict="false",
                                pointer=True, pip_fails=True)
    assert result.returncode == 0
    assert output == "enabled=false"


def test_strict_with_midway_error_fails(step_script, tmp_path):
    result, _, _ = execute(step_script, tmp_path, want="true", strict="true",
                           pointer=True, pip_fails=True)
    assert result.returncode == 1


def test_effective_value_feeds_the_env_file():
    text = DEPLOY.read_text()
    assert "steps.classifier.outputs.enabled" in text
    assert "vars.CLASSIFIER_ENABLED ||" not in text.split("Atualiza código e .env")[1][:1500]


