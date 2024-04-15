from pydantic import BaseSettings


class Settings(BaseSettings):
    app_environment = 'dev'
    model: str = 'mistral-7b-instruct-v0.1.Q4_0'
    gpt4all_path: str = '/models'
    # gpt4all_path = '/home/dell/Documents/asset_panda/gpt4all/gpt4all-api/gpt4all_api/models'
    inference_mode: str = "cpu"
    hf_inference_server_host: str = "http://gpt4all_gpu:80/generate"
    sentry_dns: str = None

    temp: float = 0.18
    top_p: float = 1.0
    top_k: int = 50
    repeat_penalty: float = 1.18



settings = Settings()
