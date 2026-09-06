# TaNERlan web playground

Минимальный интерфейс для демонстрации API разметки. Он включает mock-реализацию
`POST /api/v1/predict`, которая принимает и возвращает данные в формате
[`API.md`](../API.md), но использует предсказуемые правила вместо модели.

Запуск без Docker:

```bash
cd web
python -m pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

Запуск в Docker из корня репозитория:

```bash
docker compose -f docker-compose.web.yml up --build
```

Открыть `http://localhost:8080`. Чтобы подключить реальный сервис, сохраните UI,
а обработчик `/api/v1/predict` замените проксированием на основной API: формат
запроса и ответа уже совпадает.
