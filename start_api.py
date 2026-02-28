import uvicorn
from src.api.main import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
