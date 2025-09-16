import asyncio
from chatbot.medical_orchestrator import MedicalOrchestrator

async def test_orchestrator():
    orch = MedicalOrchestrator()
    result = await orch.process_medical_query('What is the difference between Type 1 and Type 2 diabetes?', {})
    print('Response:', result.final_response[:300])
    print('Confidence:', result.consensus_confidence)
    print('Risk Level:', result.risk_level)

if __name__ == "__main__":
    asyncio.run(test_orchestrator())