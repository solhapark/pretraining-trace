#!/bin/bash
# GitHub Personal Access Token 설정 스크립트

echo "GitHub Personal Access Token 설정"
echo "=================================="
echo ""
echo "1. GitHub에서 토큰을 생성했나요? (Settings > Developer settings > Personal access tokens)"
echo "2. 토큰을 복사했나요?"
echo ""
read -p "토큰을 입력하세요: " GITHUB_TOKEN

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: 토큰이 입력되지 않았습니다."
    exit 1
fi

# Git credential helper 설정 (store 방식)
git config --global credential.helper store

# Remote URL에 토큰 포함 (임시 방법)
# 보안상 권장하지 않지만, credential helper가 작동하지 않을 때 사용 가능
# git remote set-url origin https://${GITHUB_TOKEN}@github.com/solhapark/pretraining-trace.git

echo ""
echo "설정 완료!"
echo ""
echo "다음 명령어로 테스트하세요:"
echo "  git push"
echo ""
echo "또는 credential helper를 사용하려면:"
echo "  echo 'https://${GITHUB_TOKEN}@github.com' > ~/.git-credentials"
echo "  chmod 600 ~/.git-credentials"
