c.ServerProxy.servers = {
    "streamlit": {
        "command": [
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "{port}",
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ],
        "timeout": 120,
        "launcher_entry": {
            "title": "Cardiac MRI U-Net Streamlit Demo",
        },
    }
}
