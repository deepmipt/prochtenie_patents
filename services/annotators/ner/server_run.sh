while ! pip install -r common_requirements.txt ; do sleep 5; done

python -m deeppavlov download $DEEPPAVLOV_CONFIG

gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --reload --timeout 120