FROM python:3.10.14-bookworm
WORKDIR /app

# Only copy needed files, exclude data and results folders
COPY crp_cohort_extraction.py /app/
COPY requirements.txt /app/
COPY crp_ensemble /app/crp_ensemble

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install pandas==2.1.4
RUN pip install fhir-pyrate==0.2.1 --no-deps
RUN pip install -r requirements.txt

CMD ["python", "crp_cohort_extraction.py"]




