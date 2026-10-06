import os

# Testes nunca enviam traces ao Langfuse real, mesmo com chaves no .env.
# (load_dotenv não sobrescreve variáveis já definidas.)
os.environ["LANGFUSE_ENABLED"] = "false"
