python main.py main.gpus=[1] backend.libero_plus.suite_name=libero_spatial main.num_workers=4 \
  perception.vlm_agent.base_url=http://localhost:9000/v1/ \
  perception.qwen_grounding.base_url=http://localhost:9000/v1/ \
  perception.qwen.base_url=http://localhost:9000/v1/ \
  backend.libero_plus.start_id=1201 backend.libero_plus.end_id=2401