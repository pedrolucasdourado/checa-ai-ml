import os
import sys

# Add the current directory to PYTHONPATH so we can import from src
sys.path.append(os.getcwd())

from src.fact_check_service import FactCheckService

def main():
    print("\n" + "="*60)
    print("🤖 Checa-AI Interactive CLI")
    print("Digite a alegação para verificar ou 'exit' para sair.")
    print("="*60)

    # Initialize service
    try:
        service = FactCheckService.get_instance()
    except Exception as e:
        print(f"Erro ao inicializar o serviço: {e}")
        return

    while True:
        try:
            query = input("\n👉 Alegação: ").strip()
            
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'sair']:
                print("Encerrando... Até logo!")
                break

            print("\n🔍 Verificando...")
            result = service.verify_claim(query)

            status = result['status']
            score = result.get('score')
            narrative = result['counter_narrative']

            if status == 'matched':
                print("\n✅ [MATCHED] Encontramos checagens oficiais!")
                print(f"Score de Similaridade: {score}")
                print("-" * 30)
                print(f"{narrative}")
                print("-" * 30)
            elif status == 'abstained':
                print("\n⚪ [ABSTAINED] Nenhuma checagem oficial encontrada.")
                print(f"Score: {score}")
                print(f"Resposta: {narrative}")
            elif status == 'blocked':
                print("\n🚫 [BLOCKED] A solicitação ou a resposta foi bloqueada pelos Guardrails.")
                print(f"Motivo: Segurança/Moderação")
                print(f"Resposta: {narrative}")

        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\n❌ Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
