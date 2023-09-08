FROM python:3.9.13

WORKDIR /app/frontend

RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]
