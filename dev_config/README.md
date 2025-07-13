## 빠른 설정

```bash
# 실행 권한 부여
chmod +x dev_config/setup_env.sh

# 환경 설정 실행
./dev_config/setup_env.sh

# 환경 활성화
conda activate jumpai-drug-discovery
```

## 빠른 설정 안될 경우 -> 수동 설정
```bash
# Conda 환경 생성
conda env create -f dev_config/environment.yml

# 환경 활성화
conda activate jumpai-drug-discovery

# 추가 패키지 설치
pip install -r dev_config/requirements.txt
```

## 업데이트

### Conda 패키지인 경우: environment.yml 업데이트
```bash
# 패키지 설치
conda install <package-name>

# environment.yml 업데이트
conda env export > dev_config/environment.yml

# Git에 커밋
git add dev_config/environment.yml
git commit -m "Update conda environment: add <package-name>"
```
### pip 패키지인 경우: requirements.txt 업데이트
```bash
# 패키지 설치
pip install <package-name>

# requirements.txt 업데이트
pip freeze > dev_config/requirements.txt

# Git에 커밋
git add dev_config/requirements.txt
git commit -m "Update pip requirements: add <package-name>"
```

## 팀원 환경 동기화

다른 팀원이 환경을 업데이트했을 때:

```bash
# 최신 코드 가져오기
git pull

# 환경 업데이트
conda env update -f dev_config/environment.yml
pip install -r dev_config/requirements.txt
```

## 환경 문제 해결

### 환경이 꼬였을 때:
```bash
# 기존 환경 삭제
conda env remove -n jumpai-drug-discovery

# 새로 생성
conda env create -f dev_config/environment.yml
conda activate jumpai-drug-discovery
pip install -r dev_config/requirements.txt
```

## 주요 패키지 목록