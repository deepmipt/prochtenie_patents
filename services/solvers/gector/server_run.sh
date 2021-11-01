while ! pip install -r common_requirements.txt ; do sleep 5; done

python preload_gector.py

gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --reload