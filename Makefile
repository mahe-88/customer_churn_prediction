
PYTHON=python
STREAMLIT=streamlit
APP=deployment/app.py
REQ=deployment/requirements.txt

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQ)

run-app:
	$(STREAMLIT) run $(APP)

test:
	pytest -v tests

git-push:
	git add .
	git commit -m "Telecommunication Customer Churn Prediction project"
	git push origin main