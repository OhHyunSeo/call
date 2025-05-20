FROM continuumio/miniconda3

# 기본 작업 디렉토리 설정
WORKDIR /app
COPY . /app

# 시스템 패키지 설치
RUN apt update && apt install -y \
  ffmpeg \
  sqlite3 \
  gcc \
  g++ \
  cmake \
  make \
  libsndfile1 \
  libprotobuf-dev \
  protobuf-compiler \
  sox \
  libmagic1

# pip 최신 버전으로 업데이트
RUN pip install --upgrade pip

# Conda 환경 생성
RUN conda env create -f environment.yaml

# 환경 경로 등록
ENV PATH /opt/conda/envs/Callytics/bin:$PATH

# SQLite DB 초기화: 기존 DB 삭제 후 schema 적용 (딱 한 줄만)
RUN rm -f .db/Callytics.sqlite && \
    /opt/conda/envs/Callytics/bin/sqlite3 .db/Callytics.sqlite < src/db/sql/Schema.sql

# 앱 실행
CMD ["conda", "run", "--no-capture-output", "-n", "Callytics", "python", "main.py"]
