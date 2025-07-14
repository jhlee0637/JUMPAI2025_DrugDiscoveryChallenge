#!/bin/bash

# JUMPAI 2025 Drug Discovery Challenge 환경 설정 스크립트

echo "🧬 JUMPAI Drug Discovery 환경을 설정합니다..."

# Conda 환경 생성
echo "📦 Conda 환경 'jumpai-drug-discovery' 생성 중..."
conda env create -f environment.yml

# 환경 활성화
echo "🔄 환경 활성화 중..."
conda activate jumpai-drug-discovery

# Jupyter 커널 등록
echo "📓 Jupyter 커널 등록 중..."
python -m ipykernel install --user --name jumpai-drug-discovery --display-name "JUMPAI Drug Discovery"

echo "✅ 환경 설정 완료!"
echo ""
echo "사용법:"
echo "1. conda activate jumpai-drug-discovery"
echo "2. jupyter notebook"
echo "3. 노트북에서 'JUMPAI Drug Discovery' 커널 선택"
echo ""
echo "또는 직접 실행:"
echo "cd experiments/baseline && jupyter notebook baseline.ipynb"
